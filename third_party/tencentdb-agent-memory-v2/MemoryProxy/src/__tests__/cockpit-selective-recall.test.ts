import { afterEach, describe, expect, it, vi } from "vitest";

import { prepareCockpitQuery } from "../injection/cockpit-query.js";
import {
  compileChineseCockpitSemantics,
  extractChinesePersonTargets,
} from "../injection/cockpit-chinese-semantics.js";
import { TdaiL1RecallInjector } from "../injection/injectors/tdai-l1-recall-injector.js";
import {
  buildCockpitRetrievalPlan,
  extractCockpitNamedTargets,
} from "../injection/cockpit-retrieval-plan.js";
import { TdaiMemoryToolsInjector } from "../injection/injectors/tdai-tools-injector.js";
import {
  __resetInjectionPipelineForTests,
  getInjectionPipeline,
} from "../injection/index.js";
import type { AgentContext } from "../injection/types.js";
import type { CockpitAnswerContract } from "../injection/cockpit-answer-contract.js";
import type { TdaiMemoryConfig } from "../tdai/types.js";
import { DEFAULT_CONFIG } from "../config.js";

const REQUEST_TIME = "2026-08-26T02:30:00.000Z";

const memoryConfig: TdaiMemoryConfig = {
  enabled: true,
  endpoint: "http://memory-core:8420",
  apiKey: "test-key",
  serviceId: "configured-space",
  writeL0: true,
  recallL1: true,
  injectL2L3: true,
  l1Limit: 5,
  l2Limit: 3,
  timeoutMs: 1_000,
};

function context(query: string): AgentContext {
  return {
    messages: [{ role: "user", blocks: [{ type: "text", content: query }] }],
    requestParams: {},
    metadata: {
      protocol: "openai",
      traceId: "trace-1",
      keyId: "key-1",
      modelId: "small-model",
      stream: false,
      agentSource: "cockpit",
      requestTime: REQUEST_TIME,
      timezone: "Asia/Shanghai",
      custom: {
        session: {
          team_id: "team-1",
          user_id: "driver-1",
          agent_id: "vehicle-agent-1",
          session_id: "trip-1",
          space_id: "vehicle-space-1",
          user_key: "authenticated-driver-key",
        },
      },
    },
  };
}

const OLD_FAILURE_QUERIES = [
  "6月第一周记录的三个充电事件中，驾驶员最常去哪个充电站，共去了几次？",
  "As of July 25, when is Maya's tire inspection finally scheduled?",
  "Where should the alias 'work' resolve on September 10 and on September 16, respectively?",
];

const CHINESE_MULTI_SLOT_FINAL_QUERY =
  "通勤策略定稿后，路线选择、高架通行和收费道路这三项现在分别是什么？";

const PRIOR_TOOL_TEXT_QUERIES = [
  "截至5月10日，驾驶员工作日早上的最新路线偏好是什么？请同时说明高架和收费站约束。",
  "小雨在车上且要吃晚餐时，综合两个人在不同会话中的要求，应该选择哪家餐厅？为什么？",
  "驾驶员最常听的播客节目是什么？",
  "Across Maya's recorded charging visits, which station was used most often and how many times?",
  "What seat position does Maya prefer for long trips?",
];

function injector(): TdaiL1RecallInjector {
  return new TdaiL1RecallInjector(memoryConfig, null, 5, 3, null, {
    domainProfile: "smart-cockpit",
    timezone: "Asia/Shanghai",
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  __resetInjectionPipelineForTests();
});

describe("cockpit selective active recall", () => {
  it("routes aggregation, final-update, and multi-date omission risks", () => {
    for (const query of OLD_FAILURE_QUERIES) {
      const prepared = prepareCockpitQuery(query, {
        requestTime: REQUEST_TIME,
        timezone: "Asia/Shanghai",
      });
      expect(prepared.shouldSearchMemory, query).toBe(true);
    }
    expect(prepareCockpitQuery(OLD_FAILURE_QUERIES[0]).reasons).toContain("aggregation-frequency");
    expect(prepareCockpitQuery(OLD_FAILURE_QUERIES[1]).reasons).toContain("latest-final-update");
    expect(prepareCockpitQuery(OLD_FAILURE_QUERIES[2]).reasons).toContain("multi-time-comparison");
    expect(prepareCockpitQuery(CHINESE_MULTI_SLOT_FINAL_QUERY).reasons).toContain("latest-final-update");
  });

  it("compiles varied natural Chinese state questions into semantic recall risks", () => {
    const cases: Array<[string, string]> = [
      ["通勤规则已经敲定，现行方案中路线怎么选？高架能不能走？收费路要不要避开？", "latest-final-update"],
      ["通勤方案定下来之后，如今路线选择、高架、收费路三条规则各是什么？", "latest-final-update"],
      ["电量只剩8%，这时找补能站应该先看距离，还是休息室和评分？", "latest-final-update"],
      ["现在电只剩8%，选充电站先按离得近排，还是还按休息室、评分排？", "latest-final-update"],
      ["‘会合点’在10月15日指哪里，到了10月19日又指哪里？", "multi-time-comparison"],
      ["只看10月16日那天已经有的记录，长途腰托是几档？", "latest-final-update"],
      ["请分别报出驾驶位顾原、副驾阿禾和后排孔老师各自的空调温度。", "cross-session-synthesis"],
      ["顾原最喜欢哪位歌手？", "subject-attribute-query"],
      ["林澈办公楼门禁卡号是多少？", "subject-attribute-query"],
      ["我需要许岚的办公门禁卡号，历史没保存的话不要猜测。", "subject-attribute-query"],
      ["请核对林澈公司门禁卡号，只有办公地点信息时应回答无法确定。", "subject-attribute-query"],
      ["夜间媒体调整后，音乐类型和音量上限分别是多少？", "latest-final-update"],
      ["现在查年检：之前挪过一次、后来不要了，是否还排着时间？", "latest-final-update"],
      ["六笔补能均已完成，请按地点汇总并报排第一的站和次数。", "aggregation-frequency"],
      ["我要生效版本的通勤三字段：路线、高架、收费道路。", "latest-final-update"],
      ["车辆检查重新约完又不去了，还有有效档期吗？", "latest-final-update"],
      ["电量到了9%，找充电位置首先取决于远近还是设施评分？", "latest-final-update"],
      ["回看10月16日的有效配置，腰部支撑是第几档？", "latest-final-update"],
      ["前面那五回实际补电，按站点归拢一下，哪儿排头名？去了几回？", "aggregation-frequency"],
      ["把已经结束的五单充电按地方并起来，出现最多的点有几单？", "aggregation-frequency"],
      ["已完成补能按地点计数后，榜首是谁？后面带上准确计数。", "aggregation-frequency"],
      ["我只要真实充电事件的众数：哪座站，以及一共出现几回。", "aggregation-frequency"],
      ["当前电池8%，选站是就近先，还是继续把休息室、评分放前面？", "latest-final-update"],
      ["只有10%余电了，决定去哪充电时哪个因素优先，另外两项是否降级？", "latest-final-update"],
      ["剩10%电时请直接给选站顺序：谁在首位，休息室、评分是否仍在首位。", "latest-final-update"],
      ["按10月16日当天已经生效的内容看，长途座椅腰部支撑是多少档？", "latest-final-update"],
      ["在10月16日收盘时点查看长途座椅配置，腰部支撑显示几档？", "latest-final-update"],
      ["那五次已经充完的记录按地点算频数，最高的是哪个站，出现几次？", "aggregation-frequency"],
      ["别数导航意图，把落成的充电按站名汇总，第一位和频次都报出。", "aggregation-frequency"],
      ["把落地的五回充能按站分组，最大一组是哪座、包含几回？", "aggregation-frequency"],
      ["仪表显示8%电量，补能位置应先比较远近还是设施和评分？", "latest-final-update"],
      ["仪表为8%，选充电站的第一准则是什么，另外两项还优先吗？", "latest-final-update"],
      ["以10月16日当天结束为边界查看配置，长途座椅腰部支撑显示多少档？", "latest-final-update"],
      ["按10月16日当天已经落地的配置作答，长途腰部支撑处于几档？", "latest-final-update"],
      ["夜间媒体的新版本里，播放哪种音乐，最大音量多少，换下来的还播吗？", "latest-final-update"],
      ["查清车检安排的最终落点：现有预约、撤销时段、替代预约分别是什么状态？", "latest-final-update"],
      ["同一个碰头口令在10月15日会导向哪里，在10月19日又会导向哪里？", "multi-time-comparison"],
      ["同一集合口令在10月15日与10月19日会给出哪两个目的地？", "multi-time-comparison"],
      ["请按日期逐项列值：10月15日对应的碰头地；10月19日对应的碰头地。", "multi-time-comparison"],
      ["本次由沈舟驾驶，宁宁在副驾位，梁叔坐后排座，请按人列常用温度。", "cross-session-synthesis"],
      ["乘车人是陆青、可可、孟姨；依次对应主驾、副驾、后排，温度各多少？", "cross-session-synthesis"],
      ["三位乘员分别为程野、小葵、孔老师，座次是驾驶席、右前席、后排；请报各人的偏好温度。", "cross-session-synthesis"],
      ["乘车人为程野、小葵、孔老师，座位依次是驾驶位、右前位、后座，请列温度。", "cross-session-synthesis"],
      ["方向盘交给苏木，宁宁坐前排，魏叔坐最后一排，请逐个给温度。", "cross-session-synthesis"],
      ["记录截断在10月16日这一日时，座椅长途腰撑处于第几档？", "latest-final-update"],
      ["以不晚于10月16日为范围查询，跑长途时腰部支撑是多少档？", "latest-final-update"],
      ["电池降至8%，最近距离、休息室、评分三者谁应排最前？", "latest-final-update"],
      ["当电量是8%时，补电位置先依什么决定，原来的设施和评分怎么办？", "latest-final-update"],
      ["车只余8%电，应该先就近还是先挑有休息室且高分的站？", "latest-final-update"],
      ["余量跌到10%后，充电地点的排序口径是什么，哪些因素被降级？", "latest-final-update"],
      ["按充电地点给五个闭环事件分桶，最大的桶叫什么，有多少个？", "aggregation-frequency"],
      ["完成记录逐站聚合以后，访问量第一的补能点及数量是什么？", "aggregation-frequency"],
      ["经过多次调整，正在执行的路线原则以及两类道路限制是什么？", "latest-final-update"],
      ["只采用最新生效记录，通勤路线怎么选，两种道路各怎么处理？", "latest-final-update"],
      ["终版导航策略包含选路准则和两种道路约束，请把三项完整列出。", "latest-final-update"],
      ["同一个会合口令在10月15日和10月19日分别会导航到什么地方？", "multi-time-comparison"],
      ["五笔闭环补能流水按站归组后，最大那组对应哪个地方，有多少笔？", "aggregation-frequency"],
      ["查询10月16日这个历史切片，后续记录不要算，腰托处于几档？", "latest-final-update"],
      ["夜里听歌配置前后有变，请以最后版本回答当前类别、上限及旧类别去留。", "latest-final-update"],
      ["这一车有陆青、可可、孟姨，座次依次主驾、右前、后排，请报每人温度。", "cross-session-synthesis"],
      ["乘员名单为程野、小葵、孔老师，分别坐驾驶席、副驾席、后座，各偏好几度？", "cross-session-synthesis"],
      ["由温言驾驶，阿禾坐在副驾，杜姐位于后排，逐个说历史空调设置。", "cross-session-synthesis"],
      ["三席人员依次是许岚、小满、孟姨，对应主驾、前排、后排，温度各几度？", "cross-session-synthesis"],
      ["只凭现有对话可否确认陆青喜欢谁唱歌？音乐类别不能当作答案。", "subject-attribute-query"],
      ["年检更新链走完后是什么结果，末版档期仍有效吗，之后出现新安排没有？", "latest-final-update"],
      ["假设时间停留在10月16日，依据当日已有信息，腰部支撑应为几档？", "latest-final-update"],
      ["不使用10月16日之后的任何更新，回答该日长途座椅腰托的有效设置。", "latest-final-update"],
    ];
    for (const [query, reason] of cases) {
      const prepared = prepareCockpitQuery(query, {
        requestTime: REQUEST_TIME,
        timezone: "Asia/Shanghai",
      });
      expect(prepared.shouldSearchMemory, query).toBe(true);
      expect(prepared.reasons, query).toContain(reason);
    }
  });

  it("compiles Chinese intent by domain and excludes scalar maxima from event aggregation", () => {
    const cases: Array<[string, string, string]> = [
      ["五条完成记录按站点归并，最高频站名和计数是什么？", "charging-event", "event-frequency"],
      ["续航告急，只剩8%，找补能点先看距离还是设施评分？", "charging-priority", "conditional-priority"],
      ["时间停在10月16日，后续变化全部排除，腰托几档？", "lumbar", "cutoff-state"],
      ["主驾是苏木，旁边是宁宁，后排是魏叔，各调多少度？", "occupant-temperature", "multi-person-state"],
      ["现有别名中，固定车位、亲友住处及诊所各绑定到哪？", "navigation-alias", "multi-target-state"],
      ["夜间媒体改完后，曲风、最大音量和旧音乐状态是什么？", "media", "correction-state"],
      ["年检走到更新链末尾是否取消，有没有替代安排？", "inspection", "final-cancellation"],
      ["四笔充电都完成了，按站汇总后排第一的是谁？", "charging-event", "event-frequency"],
      ["播放方案纠正后曲风和音量是多少，纠正前曲风还有效吗？", "media", "correction-state"],
      ["按有效期判断，10月15日的会合地址和10月19日的会合地址分别是哪？", "meeting-point", "two-date-state"],
      ["10月15日集合地点在哪里，10月19日集合地点又在哪里？", "meeting-point", "two-date-state"],
      ["10月15日集合在哪里，到了10月19日会合地点改到哪儿？", "meeting-point", "two-date-state"],
      ["按10月16日当天已经生效的内容看，长途座椅腰部支撑是多少档？", "lumbar", "cutoff-state"],
      ["在10月16日收盘时点查看长途座椅配置，腰部支撑显示几档？", "lumbar", "cutoff-state"],
      ["那五次已经充完的记录按地点算频数，最高的是哪个站，出现几次？", "charging-event", "event-frequency"],
      ["把落地的五回充能按站分组，最大一组是哪座、包含几回？", "charging-event", "event-frequency"],
      ["仪表显示8%电量，补能位置应先比较远近还是设施和评分？", "charging-priority", "conditional-priority"],
      ["仪表为8%，选充电站的第一准则是什么，另外两项还优先吗？", "charging-priority", "conditional-priority"],
      ["以10月16日当天结束为边界查看配置，长途座椅腰部支撑显示多少档？", "lumbar", "cutoff-state"],
      ["按10月16日当天已经落地的配置作答，长途腰部支撑处于几档？", "lumbar", "cutoff-state"],
      ["夜间媒体的新版本里，播放哪种音乐，最大音量多少，换下来的还播吗？", "media", "correction-state"],
      ["现在还有要执行的车检预约吗？请同时交代被撤档期及有没有重新预约。", "inspection", "final-cancellation"],
      ["查清车检安排的最终落点：现有预约、撤销时段、替代预约分别是什么状态？", "inspection", "final-cancellation"],
      ["同一个碰头口令在10月15日会导向哪里，在10月19日又会导向哪里？", "meeting-point", "two-date-state"],
      ["请按日期逐项列值：10月15日对应的碰头地；10月19日对应的碰头地。", "meeting-point", "two-date-state"],
      ["同一集合口令在10月15日与10月19日会给出哪两个目的地？", "meeting-point", "two-date-state"],
      ["记录截断在10月16日这一日时，座椅长途腰撑处于第几档？", "lumbar", "cutoff-state"],
      ["以不晚于10月16日为范围查询，跑长途时腰部支撑是多少档？", "lumbar", "cutoff-state"],
      ["电池降至8%，最近距离、休息室、评分三者谁应排最前？", "charging-priority", "conditional-priority"],
      ["当电量是8%时，补电位置先依什么决定，原来的设施和评分怎么办？", "charging-priority", "conditional-priority"],
      ["车只余8%电，应该先就近还是先挑有休息室且高分的站？", "charging-priority", "conditional-priority"],
      ["余量跌到10%后，充电地点的排序口径是什么，哪些因素被降级？", "charging-priority", "conditional-priority"],
      ["电池跌至10%要补能，三个选站标准怎么排：距离、设施、评价？", "charging-priority", "conditional-priority"],
      ["按充电地点给五个闭环事件分桶，最大的桶叫什么，有多少个？", "charging-event", "event-frequency"],
      ["完成记录逐站聚合以后，访问量第一的补能点及数量是什么？", "charging-event", "event-frequency"],
      ["经过多次调整，正在执行的路线原则以及两类道路限制是什么？", "commute", "latest-state"],
      ["只采用最新生效记录，通勤路线怎么选，两种道路各怎么处理？", "commute", "latest-state"],
      ["终版导航策略包含选路准则和两种道路约束，请把三项完整列出。", "commute", "latest-state"],
      ["同一个会合口令在10月15日和10月19日分别会导航到什么地方？", "meeting-point", "two-date-state"],
      ["五笔闭环补能流水按站归组后，最大那组对应哪个地方，有多少笔？", "charging-event", "event-frequency"],
      ["查询10月16日这个历史切片，后续记录不要算，腰托处于几档？", "lumbar", "cutoff-state"],
      ["夜里听歌配置前后有变，请以最后版本回答当前类别、上限及旧类别去留。", "media", "correction-state"],
      ["这一车有陆青、可可、孟姨，座次依次主驾、右前、后排，请报每人温度。", "occupant-temperature", "multi-person-state"],
      ["乘员名单为程野、小葵、孔老师，分别坐驾驶席、副驾席、后座，各偏好几度？", "occupant-temperature", "multi-person-state"],
      ["由温言驾驶，阿禾坐在副驾，杜姐位于后排，逐个说历史空调设置。", "occupant-temperature", "multi-person-state"],
      ["三席人员依次是许岚、小满、孟姨，对应主驾、前排、后排，温度各几度？", "occupant-temperature", "multi-person-state"],
      ["年检更新链走完后是什么结果，末版档期仍有效吗，之后出现新安排没有？", "inspection", "final-cancellation"],
      ["假设时间停留在10月16日，依据当日已有信息，腰部支撑应为几档？", "lumbar", "cutoff-state"],
      ["不使用10月16日之后的任何更新，回答该日长途座椅腰托的有效设置。", "lumbar", "cutoff-state"],
      ["五笔真实补电结果分别落在哪些站，合并同站记录后最大的那组有几笔？", "charging-event", "event-frequency"],
      ["五次真正完成的补电记录请重新盘账；按站点汇并，冠军站和累计笔数各是什么？", "charging-event", "event-frequency"],
      ["盘点五件已发生的充能事项，同站合在一组后，数量峰值属于哪里、值为多少？", "charging-event", "event-frequency"],
      ["把五个充能闭环依地点装桶，最大一桶对应哪站，桶内有几个？", "charging-event", "event-frequency"],
      ["当前续航电量10%，选充能站由哪个条件主导，默认两项有没有让位？", "charging-priority", "conditional-priority"],
      ["在余电8%的情况下，充电位置的头号选择因素是哪项，其余两项状态如何？", "charging-priority", "conditional-priority"],
      ["把历史定格在10月16日收尾，那时长途座椅的腰部支撑生效为多少档？", "lumbar", "cutoff-state"],
      ["按10月16日日终已有记录取值，之后更新不得影响长途支撑结果。", "lumbar", "cutoff-state"],
      ["请把10月16日之后的变更全部排除，那一天有效的长途腰部支撑是多少档？", "lumbar", "cutoff-state"],
      ["时间线停在10月16日，请还原那时已生效的长途座椅腰部支撑档位。", "lumbar", "cutoff-state"],
      ["不读取晚于10月16日的任何设置，按该日状态回答长途模式腰撑几档。", "lumbar", "cutoff-state"],
      ["读取现行导航策略：怎样挑上班路线，两类特殊道路各是允许还是规避？", "commute", "latest-state"],
      ["车内夜间听歌规则调整过，现行内容与最高音量分别是什么，先前内容还有效吗？", "media", "correction-state"],
    ];
    for (const [query, domain, intent] of cases) {
      const compiled = compileChineseCockpitSemantics(query);
      expect(compiled.domain, query).toBe(domain);
      expect(compiled.intents, query).toContain(intent);
      expect(prepareCockpitQuery(query).shouldSearchMemory, query).toBe(true);
    }

    const scalar = prepareCockpitQuery("夜间媒体更新后，现用什么曲风，音量最多开到多少？");
    expect(scalar.reasons).toContain("latest-final-update");
    expect(scalar.reasons).not.toContain("aggregation-frequency");

    const stationPolicy = "在电量降至8%这个条件下，哪个站点属性成为主导，原先两项还排第一吗？";
    expect(compileChineseCockpitSemantics(stationPolicy).intents).toContain("conditional-priority");
    expect(compileChineseCockpitSemantics(stationPolicy).intents).not.toContain("event-frequency");
    expect(prepareCockpitQuery(stationPolicy).reasons).not.toContain("aggregation-frequency");

    const splitDateValues = "不要只给最终值，请列出10月15日和10月19日各自有效的会面地点。";
    expect(compileChineseCockpitSemantics(splitDateValues).intents).toContain("two-date-state");
    expect(compileChineseCockpitSemantics(splitDateValues).intents).not.toContain("latest-state");
    expect(prepareCockpitQuery(splitDateValues).reasons).toContain("multi-time-comparison");
    expect(prepareCockpitQuery(splitDateValues).reasons).not.toContain("latest-final-update");
  });

  it("compiles colloquial Chinese historical-snapshot wording as a cutoff", () => {
    const queries = [
      "只按9月12日以前已经发生的更新，腰托应设几档？",
      "我要9月12日当天的历史快照，不看未来修改，腰托是什么档？",
      "回看9月12日的有效配置，长途腰托是第几档？",
      "假定日期停在9月12日，根据那一刻的更新，腰托显示几档？",
      "不要用9月12日以后的最终值，回答该日期生效的腰托档位。",
      "按10月16日当天已经生效的内容看，长途座椅腰部支撑是多少档？",
      "在10月16日收盘时点查看长途座椅配置，腰部支撑显示几档？",
      "以10月16日当天结束为边界查看配置，长途座椅腰部支撑显示多少档？",
      "按10月16日当天已经落地的配置作答，长途腰部支撑处于几档？",
      "记录截断在10月16日这一日时，座椅长途腰撑处于第几档？",
      "以不晚于10月16日为范围查询，跑长途时腰部支撑是多少档？",
      "查询10月16日这个历史切片，后续记录不要算，腰托处于几档？",
      "假设时间停留在10月16日，依据当日已有信息，腰部支撑应为几档？",
      "不使用10月16日之后的任何更新，回答该日长途座椅腰托的有效设置。",
      "把历史定格在10月16日收尾，那时长途座椅的腰部支撑生效为多少档？",
      "按10月16日日终已有记录取值，之后更新不得影响长途支撑结果。",
      "请把10月16日之后的变更全部排除，那一天有效的长途腰部支撑是多少档？",
      "时间线停在10月16日，请还原那时已生效的长途座椅腰部支撑档位。",
      "不读取晚于10月16日的任何设置，按该日状态回答长途模式腰撑几档。",
      "将历史定格到10月16日当天，长途行驶对应的腰部支撑配置是什么？",
      "后于10月16日的更新不要算，查询该日已经有效的长途腰托档数。",
      "回到10月16日的状态快照，忽略未来变化，当时长途座椅调在几档？",
    ];
    for (const query of queries) {
      const compiled = compileChineseCockpitSemantics(query);
      expect(compiled.intents, query).toContain("cutoff-state");
      expect(prepareCockpitQuery(query).reasons, query).toContain("latest-final-update");
    }
  });

  it("routes Chinese stored-memory and missing-evidence questions to bounded recall", () => {
    const queries = [
      "只依据已存记忆，能确定林川最喜欢的歌手姓名吗？不能就明确说不能。",
      "记忆里明确存过周宁的办公楼卡号吗？请勿编造数字。",
      "请核实沈禾的办公门禁卡号，证据没覆盖这个字段时直接拒答。",
      "记录里有没有江屿办公楼门卡编号？没有的话限定字段拒答。",
      "现有对话能提供程野办公门卡的准确编号吗？",
      "江屿办公楼门卡ID是多少？记录没覆盖就明确说不知道。",
      "以前是否记录了温言偏爱谁演唱？泛化的音乐风格不算人物证据。",
      "车载对话足以回答江屿的办公证件号码吗？找不到直接值就限定字段拒答。",
      "请从已有记录返回程野工作门卡卡号，不能拿地点或公司名称代替。",
      "核实许岚办公卡的具体编号，若历史没有记录便不要补全。",
    ];
    for (const query of queries) {
      const prepared = prepareCockpitQuery(query);
      expect(prepared.shouldSearchMemory, query).toBe(true);
      expect(prepared.reasons, query).toEqual(expect.arrayContaining([
        "history-reference",
        "subject-attribute-query",
      ]));
    }
  });

  it("routes a company access-card field check even when only wrong-field context is mentioned", () => {
    const prepared = prepareCockpitQuery(
      "请核对林澈公司门禁卡号，只有办公地点信息时应回答无法确定。",
    );
    expect(prepared.shouldSearchMemory).toBe(true);
    expect(prepared.reasons).toContain("subject-attribute-query");
  });

  it("extracts names and roles from three common Chinese occupant word orders", () => {
    const cases = [
      "主驾顾原、副驾阿禾、后排孔老师各自想要多少度？",
      "林澈在驾驶位、豆豆在副驾、杜姐在后座时分别几度？",
      "当前三位乘员的温度：温言、阿禾、杜姐各多少度？",
      "综合以前不同会话，陆青开车、可可坐右前、孟姨坐后面时，各自温度是多少？",
      "请列一个三人温度表，程野主驾、小葵副驾、孔老师后排分别对应几度。",
      "三席分开回答：江宁开车，阿澄坐右边，魏叔在后排，各设几度？",
      "温度表按归属列：司机乔木、副驾驶朵朵、后排乘客梁姨各是多少？",
      "驾驶席是林川，右前席是可心，后座是孟叔，三处温度怎么设？",
      "请列温度：沈禾对应主驾，豆豆对应副驾，方姨对应后排。",
      "保持人物归属：许青主驾、小满前排、周叔后排各几度？",
      "顾原负责开车，阿禾坐右边，孔老师坐后面，三个人的空调各调几度？",
      "车里是周遥、小满、魏叔三个人，请按各自历史报温度。",
      "开车的是沈舟，右边坐宁宁，后面坐梁叔，请逐人报空调偏好。",
      "车上三人：周遥、小满、魏叔，各自历史温度是多少？",
      "乘员名单：许岚、小满、孟姨，按三人的独立记录列温度。",
      "本次由沈舟驾驶，宁宁在副驾位，梁叔坐后排座，请按人列常用温度。",
      "乘车人是陆青、可可、孟姨；依次对应主驾、副驾、后排，温度各多少？",
      "三位乘员分别为程野、小葵、孔老师，座次是驾驶席、右前席、后排；请报各人的偏好温度。",
      "方向盘交给苏木，宁宁坐前排，魏叔坐最后一排，请逐个给温度。",
      "开车人温言，旁边阿禾，后排杜姐，分别调到几度才符合各自记录？",
      "乘车人为程野、小葵、孔老师，座位依次是驾驶位、右前位、后座，请列温度。",
      "由温言开车，阿禾在副驾驶，杜姐在后排，分别偏好车内几度？",
      "司机江屿、副驾驶豆豆、后排乘客梁叔同车时，三套空调温度如何设置？",
      "这一车有陆青、可可、孟姨，座次依次主驾、右前、后排，请报每人温度。",
      "乘员名单为程野、小葵、孔老师，分别坐驾驶席、副驾席、后座，各偏好几度？",
      "由温言驾驶，阿禾坐在副驾，杜姐位于后排，逐个说历史空调设置。",
      "三席人员依次是许岚、小满、孟姨，对应主驾、前排、后排，温度各几度？",
      "顾原负责开车，阿禾坐右前排，孔老师坐后座，请逐人报常用空调温度。",
      "方向盘由陆青掌握，可可在旁边，孟姨坐后面，请按人物查询温度。",
      "车内三席人员按前后顺序为程野、小葵、孔老师，座次主驾、副驾、后排，各用几度？",
      "请查江屿、豆豆、梁叔各自的历史温度，他们依次在驾驶位、前排、后排。",
      "请查询林澈、豆豆、杜姐三人的温度偏好，他们依次坐主驾、副驾和后座。",
      "本次座次为驾驶席沈舟、前排右席宁宁、后排梁叔，每个人常设几度？",
      "方向盘交给程野，小葵位于副驾驶席，孔老师乘坐后座；分别查温度习惯。",
      "三位乘员温言、阿禾、杜姐依次在驾驶位、前排右侧、后座，他们各用几度？",
      "驾驶位上的许岚、右侧前排的小满和后座的孟姨，各自偏好的空调设定是多少？",
    ];
    const expected = [
      ["顾原", "阿禾", "孔老师"],
      ["林澈", "豆豆", "杜姐"],
      ["温言", "阿禾", "杜姐"],
      ["陆青", "可可", "孟姨"],
      ["程野", "小葵", "孔老师"],
      ["江宁", "阿澄", "魏叔"],
      ["乔木", "朵朵", "梁姨"],
      ["林川", "可心", "孟叔"],
      ["沈禾", "豆豆", "方姨"],
      ["许青", "小满", "周叔"],
      ["顾原", "阿禾", "孔老师"],
      ["周遥", "小满", "魏叔"],
      ["沈舟", "宁宁", "梁叔"],
      ["周遥", "小满", "魏叔"],
      ["许岚", "小满", "孟姨"],
      ["沈舟", "宁宁", "梁叔"],
      ["陆青", "可可", "孟姨"],
      ["程野", "小葵", "孔老师"],
      ["苏木", "宁宁", "魏叔"],
      ["温言", "阿禾", "杜姐"],
      ["程野", "小葵", "孔老师"],
      ["温言", "阿禾", "杜姐"],
      ["江屿", "豆豆", "梁叔"],
      ["陆青", "可可", "孟姨"],
      ["程野", "小葵", "孔老师"],
      ["温言", "阿禾", "杜姐"],
      ["许岚", "小满", "孟姨"],
      ["顾原", "阿禾", "孔老师"],
      ["陆青", "可可", "孟姨"],
      ["程野", "小葵", "孔老师"],
      ["江屿", "豆豆", "梁叔"],
      ["林澈", "豆豆", "杜姐"],
      ["沈舟", "宁宁", "梁叔"],
      ["程野", "小葵", "孔老师"],
      ["温言", "阿禾", "杜姐"],
      ["许岚", "小满", "孟姨"],
    ];
    cases.forEach((query, index) => {
      expect(extractChinesePersonTargets(query).map((item) => item.name)).toEqual(expected[index]);
      expect(prepareCockpitQuery(query).reasons).toContain("cross-session-synthesis");
    });

    const positionConnectors = "陆青位于驾驶位、可可位于前排右席、孟姨位于后座，各自温度习惯怎样？";
    expect(extractChinesePersonTargets(positionConnectors).map((item) => item.name)).toEqual(["陆青", "可可", "孟姨"]);
    expect(compileChineseCockpitSemantics(positionConnectors).people).toEqual([
      { name: "陆青", role: "driver", roleLabel: "驾驶位" },
      { name: "可可", role: "front-passenger", roleLabel: "前排右席" },
      { name: "孟姨", role: "rear-passenger", roleLabel: "后座" },
    ]);
  });

  it("does not turn a count-before-list phrase into an extra named target", () => {
    expect(extractCockpitNamedTargets(
      "车载导航现有三项映射是什么，依次回答固定车位、亲友住处、诊所。",
    )).toEqual(["固定车位", "亲友住处", "诊所"]);
    expect(extractCockpitNamedTargets(
      "现在喊固定车位、亲友住处、诊所时，导航会分别解析到哪里？",
    )).toEqual(["固定车位", "亲友住处", "诊所"]);
    expect(extractCockpitNamedTargets(
      "查清车检安排的最终落点：现有预约、撤销时段、替代预约分别是什么状态？",
    )).toEqual([]);
    expect(extractCockpitNamedTargets(
      "经过多次调整，正在执行的路线原则以及两类道路限制是什么？",
    )).toEqual(["路线原则", "高架", "收费道路"]);
    expect(extractCockpitNamedTargets(
      "只采用最新生效记录，通勤路线怎么选，两种道路各怎么处理？",
    )).toEqual(["路线", "高架", "收费道路"]);
    expect(extractCockpitNamedTargets(
      "终版导航策略包含选路准则和两种道路约束，请把三项完整列出。",
    )).toEqual(["选路准则", "高架", "收费道路"]);
    expect(extractCockpitNamedTargets(
      "车机现在听到“固定车位”“家人住所”“就诊地点”会分别导航去哪？",
    )).toEqual(["固定车位", "亲友住处", "诊所"]);
    expect(extractCockpitNamedTargets(
      "查车载记忆里的三条现行别名关系：停车位、亲友家、诊所分别是什么？",
    )).toEqual(["固定车位", "亲友住处", "诊所"]);
    expect(extractCockpitNamedTargets(
      "车内夜间听歌规则调整过，现行内容与最高音量分别是什么，先前内容还有效吗？",
    )).toEqual(["音乐类型", "音量上限"]);
    expect(extractCockpitNamedTargets(
      "读取现行导航策略：怎样挑上班路线，两类特殊道路各是允许还是规避？",
    )).toEqual(["路线", "高架", "收费道路"]);
    expect(extractCockpitNamedTargets(
      "只有8%余电要进充电站，远近、休息室、口碑的优先级分别是什么？",
    )).toEqual(["距离", "休息室", "评分"]);
    for (const query of [
      "请查地点别名的末版绑定：惯用车位、亲属住址、门诊位置分别指向哪里？",
      "当前说常用车位、家属家、诊疗点时，导航分别会落到哪个地址？",
      "核对三项有效地点口令：固定停车处对应哪里，亲人住处哪里，门诊点哪里？",
      "不要漏项，列出车载记忆中惯常车位、亲属住处、就医点各自的现行目的地。",
      "请将常停地点、家人住址、看诊处与各自当前地址一一配对。",
      "查询最新版地点映射表，只需这三项：固定车位、亲属住所、门诊地点。",
      "现行地点映射中，常停车点、家人住处和看诊地点各对应何处？",
      "三处常用地点最后如何配对，惯常停车点、亲属住所、就医点分别为何？",
    ]) {
      expect(extractCockpitNamedTargets(query), query).toEqual(["固定车位", "亲友住处", "诊所"]);
      const compiled = compileChineseCockpitSemantics(query);
      expect(compiled.domain, query).toBe("navigation-alias");
      expect(compiled.intents, query).toEqual(expect.arrayContaining(["latest-state", "multi-target-state"]));
    }
  });

  it("positionally binds an explicitly ordered Chinese name list to its seats", () => {
    for (const [query, rearLabel] of [
      ["三位乘员为沈舟、宁宁、梁叔，座位依次驾驶席、前排右侧、后排，温度分别多少？", "后排"],
      ["车内三席人员依次为沈舟、宁宁、梁叔，对应驾驶席、前排右侧、第二排，各自几度？", "第二排"],
    ]) {
      const people = extractChinesePersonTargets(query);

      expect(people, query).toEqual([
        { name: "沈舟", role: "driver", roleLabel: "驾驶席" },
        { name: "宁宁", role: "front-passenger", roleLabel: "前排右侧" },
        { name: "梁叔", role: "rear-passenger", roleLabel: rearLabel },
      ]);

      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query));
      expect(plan.queries.map((item) => item.text), query).toEqual(expect.arrayContaining([
        "[沈舟] 我是驾驶员时 座舱空调偏好",
        "[宁宁] 我是副驾时 座舱空调偏好",
        "[梁叔] 我是后排乘客时 座舱空调偏好",
      ]));
    }
  });

  it("keeps vehicle-prefixed seat-manifest nouns out of occupant names", () => {
    for (const query of [
      "车内席位表为顾原主驾座位、阿禾副驾席位、孔老师第二排乘员，请分别查询空调温度。",
      "本车座位表列为顾原主驾座位、阿禾副驾席位、孔老师第二排乘员，请逐人报温度。",
      "座舱席位表显示为顾原主驾座位、阿禾副驾席位、孔老师第二排乘员，三人各几度？",
    ]) {
      expect(extractChinesePersonTargets(query), query).toEqual([
        { name: "顾原", role: "driver", roleLabel: "主驾座位" },
        { name: "阿禾", role: "front-passenger", roleLabel: "副驾席位" },
        { name: "孔老师", role: "rear-passenger", roleLabel: "第二排乘员" },
      ]);
      expect(compileChineseCockpitSemantics(query).intents, query).toContain("multi-person-state");
    }
  });

  it("compiles productive media content nouns into a complete correction contract", () => {
    for (const query of [
      "夜间播放规则修订版已生效，新音乐门类、音量封顶、旧播放去留分别怎样？",
      "晚间音频定稿后，现行媒体类别、声音上限、原播放效力各是什么？",
      "夜听配置改完，新音频类型、声量封顶、老音乐状态分别如何？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.domain, query).toBe("media");
      expect(semantics.targets.map((item) => item.key), query).toEqual(["music-type", "volume-limit"]);
      expect(semantics.intents, query).toEqual(expect.arrayContaining([
        "latest-state", "multi-target-state", "correction-state",
      ]));
      expect(extractCockpitNamedTargets(query), query).toHaveLength(2);
    }
  });

  it("does not confuse a rear-row phrase with a latest-state request", () => {
    const prepared = prepareCockpitQuery(
      "方向盘交给苏木，宁宁坐前排，魏叔坐最后一排，请逐个给温度。",
    );
    expect(prepared.reasons).toContain("cross-session-synthesis");
    expect(prepared.reasons).not.toContain("latest-final-update");
  });

  it("compiles colloquial seat-support cutoff wording into a bounded lumbar timeline", () => {
    const query = "回到10月16日当日状态，过滤未来更新后，长途座椅支撑值为第几档？";
    const semantics = compileChineseCockpitSemantics(query);
    const prepared = prepareCockpitQuery(query, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    });
    const plan = buildCockpitRetrievalPlan(query, prepared);

    expect(semantics.domain).toBe("lumbar");
    expect(semantics.intents).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
    expect(semantics.canonicalRetrievalQueries).toEqual(expect.arrayContaining([
      expect.stringContaining("座椅支撑"),
    ]));
    expect(plan.cutoffDate).toBe("2026-10-16");
    expect(plan.queries.map((item) => item.text).join("\n")).toContain("座椅支撑");
  });

  it("compiles spoken Chinese meeting, ordering, and office-card instructions", () => {
    const meeting = "两个时点分别作答：10月15日在哪里碰面，10月19日在哪里碰面？";
    const meetingSemantics = compileChineseCockpitSemantics(meeting);
    const meetingPrepared = prepareCockpitQuery(meeting);
    const meetingPlan = buildCockpitRetrievalPlan(meeting, meetingPrepared);
    expect(meetingSemantics.domain).toBe("meeting-point");
    expect(meetingSemantics.intents).toContain("two-date-state");
    expect(meetingSemantics.targets.map((item) => item.key)).toContain("meeting-point");
    expect(meetingPrepared.reasons).toContain("multi-time-comparison");
    expect(meetingPlan.highRisk).toBe(true);
    expect(meetingPlan.queries.map((item) => item.text).join("\n")).toContain("临时有效期");

    const ordering = "只剩8%电量时哪个选站因素应置顶，休息条件和高评分是否退后？";
    const orderingSemantics = compileChineseCockpitSemantics(ordering);
    const orderingPrepared = prepareCockpitQuery(ordering);
    const orderingPlan = buildCockpitRetrievalPlan(ordering, orderingPrepared);
    expect(orderingSemantics.domain).toBe("charging-priority");
    expect(orderingSemantics.intents).toContain("conditional-priority");
    expect(orderingPrepared.reasons).toContain("latest-final-update");
    expect(orderingPlan.highRisk).toBe(true);

    const officeCard = "核实许岚办公卡的具体编号，若历史没有记录便不要补全。";
    const officePrepared = prepareCockpitQuery(officeCard);
    const officePlan = buildCockpitRetrievalPlan(officeCard, officePrepared);
    expect(officePrepared.shouldSearchMemory).toBe(true);
    expect(officePrepared.reasons).toContain("subject-attribute-query");
    expect(officePlan.highRisk).toBe(true);
  });

  it("compiles productive Chinese boundary, counting, seat, and card paraphrases", () => {
    const cutoffQueries = [
      "让时间线只走到10月16日日终，按那时有效记录，长途座椅腰部支撑是几档？",
      "把历史上限设在10月16日，不纳入后来变更，当天长途腰托处于哪一档？",
      "将记忆冻结到10月16日收尾，查询那个快照里的长途座椅腰托设置。",
      "只开放10月16日及此前的记录，求该日期有效的长途模式腰托档位。",
      "时间边界卡在10月16日当天时，长途驾驶对应的腰部支撑配置是什么？",
    ];
    for (const query of cutoffQueries) {
      const semantics = compileChineseCockpitSemantics(query);
      const prepared = prepareCockpitQuery(query, {
        requestTime: "2026-10-20T01:00:00.000Z",
        timezone: "Asia/Shanghai",
      });
      const plan = buildCockpitRetrievalPlan(query, prepared);
      expect(semantics.domain, query).toBe("lumbar");
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
      expect(prepared.reasons, query).toContain("latest-final-update");
      expect(plan.highRisk, query).toBe(true);
      expect(plan.cutoffDate, query).toBe("2026-10-16");
    }

    for (const query of [
      "请把五条落地补电流水归到各自场站，哪处数量最大，具体有几条？",
      "五次确实发生的充能按地点计票，票数第一的是哪里，得到几票？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const prepared = prepareCockpitQuery(query);
      expect(semantics.domain, query).toBe("charging-event");
      expect(semantics.intents, query).toContain("event-frequency");
      expect(prepared.reasons, query).toContain("aggregation-frequency");
      expect(buildCockpitRetrievalPlan(query, prepared).highRisk, query).toBe(true);
    }

    const meeting = compileChineseCockpitSemantics(
      "按时点拆开回答：10月15日碰面在哪儿，10月19日碰面又在哪儿？",
    );
    expect(meeting.domain).toBe("meeting-point");
    expect(meeting.targets.map((item) => item.key)).toContain("meeting-point");

    const peopleQuery = "车内座次从前到后为陆青主驾、可可副驾、孟姨后座，请保持人物归属回答。";
    expect(extractChinesePersonTargets(peopleQuery)).toEqual([
      { name: "陆青", role: "driver", roleLabel: "主驾" },
      { name: "可可", role: "front-passenger", roleLabel: "副驾" },
      { name: "孟姨", role: "rear-passenger", roleLabel: "后座" },
    ]);
    const peopleSemantics = compileChineseCockpitSemantics(peopleQuery);
    const peoplePrepared = prepareCockpitQuery(peopleQuery);
    expect(peopleSemantics.domain).toBe("occupant-temperature");
    expect(peopleSemantics.intents).toContain("multi-person-state");
    expect(peoplePrepared.reasons).toContain("cross-session-synthesis");

    const priorityQuery = "电池还剩8%的紧急状态下，哪个站点因素置首，其余两项还在前列吗？";
    const prioritySemantics = compileChineseCockpitSemantics(priorityQuery);
    expect(prioritySemantics.domain).toBe("charging-priority");
    expect(prioritySemantics.intents).toContain("conditional-priority");

    const cardQuery = "车载历史有没有程野的员工通行卡编号？证据不到号码级就答未知。";
    const cardPrepared = prepareCockpitQuery(cardQuery);
    expect(cardPrepared.reasons).toContain("subject-attribute-query");
    expect(buildCockpitRetrievalPlan(cardQuery, cardPrepared)).toMatchObject({
      highRisk: true,
      risks: expect.arrayContaining(["subject-attribute-query"]),
    });
  });

  it("compiles productive Chinese idioms into complete domain contracts", () => {
    for (const query of [
      "把记忆的停止线放在10月16日日终，按这个边界查长途腰托档位。",
      "将查询时钟锁到10月16日当天，读取那个时间切片里的长途腰撑配置。",
      "把检索截止线设在10月16日，查询当时的长途座椅支撑。",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query, {
        requestTime: "2026-10-20T01:00:00.000Z",
        timezone: "Asia/Shanghai",
      }));
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
      expect(plan.cutoffDate, query).toBe("2026-10-16");
    }

    const frequency = "五趟充电都结束后按地点合账，哪座站占了多数，共有几趟？";
    const frequencySemantics = compileChineseCockpitSemantics(frequency);
    const frequencyPlan = buildCockpitRetrievalPlan(frequency, prepareCockpitQuery(frequency));
    expect(frequencySemantics.intents).toContain("event-frequency");
    expect(frequencyPlan).toMatchObject({ highRisk: true, minEvidence: 5 });

    for (const query of [
      "请查询林澈工作通行证的编号，没有直接字段就回答无法确定。",
      "我只要周遥公司出入凭证ID，记忆未保存时不要补造。",
      "核实程野员工门卡序号，记录没有这个字段就拒答。",
      "某人办公通行证号需要明确记录，证据不足时请拒答。",
    ]) {
      const prepared = prepareCockpitQuery(query);
      expect(prepared.reasons, query).toContain("subject-attribute-query");
      expect(buildCockpitRetrievalPlan(query, prepared).highRisk, query).toBe(true);
    }

    for (const query of [
      "仪表显示8%电量，挑充电点谁坐头把交椅，休息室与口碑呢？",
      "续航剩下10%时，哪项排在首位，设施条件和高评分是否让位？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.domain, query).toBe("charging-priority");
      expect(semantics.intents, query).toContain("conditional-priority");
      expect(prepareCockpitQuery(query).reasons, query).toContain("latest-final-update");
    }

    for (const query of [
      "10月15日相约在哪，10月19日相约在哪？",
      "给集合地做双日期快照：10月15日和10月19日各是什么地方？",
      "10月15日碰头在哪、10月19日碰头在哪？",
      "请返回10月15日与10月19日的约见位置。",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.domain, query).toBe("meeting-point");
      expect(semantics.targets.map((item) => item.key), query).toContain("meeting-point");
    }

    const occupants = "本次程野负责驾驶，小葵乘右前，孔老师乘后排，空调偏好逐个回答。";
    expect(extractChinesePersonTargets(occupants)).toEqual([
      { name: "程野", role: "driver", roleLabel: "主驾" },
      { name: "小葵", role: "front-passenger", roleLabel: "副驾" },
      { name: "孔老师", role: "rear-passenger", roleLabel: "后排" },
    ]);

    for (const query of [
      "多轮调整后真正有效的通勤配置是什么，三项规则都不能遗漏。",
      "核对车载导航末版上班规则，请完整列出选路及道路限制。",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.targets.map((item) => item.key), query).toEqual([
        "route", "elevated-road", "toll-road",
      ]);
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "multi-target-state"]));
    }

    const aliases = compileChineseCockpitSemantics(
      "导航绑定收口以后，车位、家属住所、门诊点三项分别在哪里？",
    );
    expect(aliases.targets.map((item) => item.key)).toEqual([
      "parking-alias", "relative-home-alias", "clinic-alias",
    ]);
    expect(aliases.intents).toEqual(expect.arrayContaining(["latest-state", "multi-target-state"]));
  });

  it("grounds unseen Chinese meeting, commute, alias, and occupant surfaces", async () => {
    const scenarios: Array<{
      query: string;
      evidence: Array<Record<string, unknown>>;
      expectedFacts: Array<{ label: string; value: string }>;
    }> = [
      {
        query: "两个时点不要合并，10月15日相约在哪，10月19日相约在哪？",
        evidence: [
          { id: "meeting-base", type: "alias", content: "10月4日平时‘会合点’指云港商务区。", updated_at: "2026-10-04T10:00:00+08:00", score: 0.8 },
          { id: "meeting-temp", type: "alias", content: "10月12日至10月17日，‘会合点’临时改为青石研发院。", updated_at: "2026-10-09T10:00:00+08:00", score: 0.9 },
          { id: "meeting-restore", type: "alias", content: "10月18日起把‘会合点’恢复到云港商务区。", updated_at: "2026-10-18T06:30:00+08:00", score: 0.95 },
        ],
        expectedFacts: [
          { label: "10月15日", value: "青石研发院" },
          { label: "10月19日", value: "云港商务区" },
        ],
      },
      {
        query: "多轮调整后真正有效的通勤配置是什么，三项规则都不能遗漏。",
        evidence: [
          { id: "commute-old", type: "preference", content: "通勤先按红绿灯最少的路线走，高架和收费道路都绕开。", updated_at: "2026-10-03T07:00:00+08:00", score: 0.8 },
          { id: "commute-final", type: "preference", content: "最终通勤选择里程最短路线；高架可以通行；收费道路继续避开。", updated_at: "2026-10-12T07:00:00+08:00", score: 0.95 },
        ],
        expectedFacts: [
          { label: "路线选择", value: "最终通勤选择里程最短路线" },
          { label: "高架", value: "高架可以通行" },
          { label: "收费道路", value: "收费道路继续避开" },
        ],
      },
      {
        query: "导航绑定收口以后，车位、家属住所、门诊点三项分别在哪里？",
        evidence: [
          { id: "alias-old", type: "alias", content: "此前记录过三个导航别名。", updated_at: "2026-10-08T10:00:00+08:00", score: 0.8 },
          { id: "alias-final", type: "alias", content: "统一更新导航绑定，三项当前映射为清溪中心车库、听泉路279号、安澜诊所19室。", updated_at: "2026-10-18T10:00:00+08:00", score: 0.95 },
        ],
        expectedFacts: [
          { label: "固定车位", value: "清溪中心车库" },
          { label: "亲友住处", value: "听泉路279号" },
          { label: "诊所", value: "安澜诊所19室" },
        ],
      },
      {
        query: "本次程野负责驾驶，小葵乘右前，孔老师乘后排，空调偏好逐个回答。",
        evidence: [
          { id: "driver", type: "preference", content: "[程野] 我是驾驶员时，座舱空调偏好22度。", score: 0.9 },
          { id: "front", type: "preference", content: "[小葵] 我是副驾时，座舱空调偏好26度。", score: 0.9 },
          { id: "rear", type: "preference", content: "[孔老师] 我是后排乘客时，座舱空调偏好24度。", score: 0.9 },
        ],
        expectedFacts: [
          { label: "主驾程野", value: "22度" },
          { label: "副驾小葵", value: "26度" },
          { label: "后排孔老师", value: "24度" },
        ],
      },
    ];

    for (const scenario of scenarios) {
      vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
        code: 0,
        data: url.endsWith("/atomic/search") ? { items: scenario.evidence } : { messages: [] },
      }), { status: 200 })));
      const contract = (await injector().execute(context(scenario.query)))[0]?.metadata?.cockpitAnswerContract as CockpitAnswerContract | undefined;
      expect(contract?.sufficient, scenario.query).toBe(true);
      expect(contract?.requiredFacts, scenario.query).toEqual(expect.arrayContaining(scenario.expectedFacts));
      vi.unstubAllGlobals();
    }
  });

  it("compiles natural Chinese role, boundary, priority, and final-state surfaces", () => {
    const alias = compileChineseCockpitSemantics(
      "导航目前把常停车的地方、亲人家和看门诊的地方分别绑定到哪里？",
    );
    expect(alias.targets.map((item) => item.key)).toEqual([
      "parking-alias", "relative-home-alias", "clinic-alias",
    ]);
    expect(alias.intents).toEqual(expect.arrayContaining(["latest-state", "multi-target-state"]));

    expect(extractChinesePersonTargets(
      "驾驶位安排沈舟，副驾驶位安排宁宁，后排安排梁叔，分别习惯几度？",
    )).toEqual([
      { name: "沈舟", role: "driver", roleLabel: "驾驶位" },
      { name: "宁宁", role: "front-passenger", roleLabel: "副驾驶位" },
      { name: "梁叔", role: "rear-passenger", roleLabel: "后排" },
    ]);
    expect(extractChinesePersonTargets(
      "开车的是程野，副驾乘员是小葵，后排乘员是孔老师，请逐人回答空调设置。",
    )).toEqual([
      { name: "程野", role: "driver", roleLabel: "主驾" },
      { name: "小葵", role: "front-passenger", roleLabel: "副驾乘员" },
      { name: "孔老师", role: "rear-passenger", roleLabel: "后排乘员" },
    ]);

    const priority = compileChineseCockpitSemantics(
      "仪表仅有8%电，选充电点最要紧的因素是什么，休息室与口碑怎么排？",
    );
    expect(priority.domain).toBe("charging-priority");
    expect(priority.intents).toContain("conditional-priority");

    const meeting = compileChineseCockpitSemantics(
      "10月15日当天相约何处，10月19日当天又相约何处？两项都要给出。",
    );
    expect(meeting.domain).toBe("meeting-point");
    expect(meeting.targets.map((item) => item.key)).toEqual(["meeting-point"]);

    const cutoffQuery = "以10月16日为历史上限，不采用后来记录，给出那天的长途腰托档数。";
    const cutoff = compileChineseCockpitSemantics(cutoffQuery);
    const cutoffPlan = buildCockpitRetrievalPlan(cutoffQuery, prepareCockpitQuery(cutoffQuery, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    }));
    expect(cutoff.intents).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
    expect(cutoffPlan).toMatchObject({ highRisk: true, cutoffDate: "2026-10-16" });

    const commute = compileChineseCockpitSemantics(
      "通勤导航最后落地的三条设置请展开说，不能漏掉任何一种道路约束。",
    );
    expect(commute.targets.map((item) => item.key)).toEqual([
      "route", "elevated-road", "toll-road",
    ]);

    const media = compileChineseCockpitSemantics(
      "核对夜间媒体终稿，要求同时回答播放种类、音量天花板和原种类是否有效。",
    );
    expect(media.targets.map((item) => item.key)).toEqual(["music-type", "volume-limit"]);
    expect(media.intents).toEqual(expect.arrayContaining([
      "latest-state", "multi-target-state", "correction-state",
    ]));
  });

  it("compiles Chinese composite seat roles, historical boundaries, and named-singer fields", () => {
    expect(extractChinesePersonTargets(
      "司机周遥、副驾乘客小满、后座乘客魏叔各自习惯的温度是什么？",
    )).toEqual([
      { name: "周遥", role: "driver", roleLabel: "司机" },
      { name: "小满", role: "front-passenger", roleLabel: "副驾乘客" },
      { name: "魏叔", role: "rear-passenger", roleLabel: "后座乘客" },
    ]);

    const cutoffQueries = [
      "将10月16日视为最后可用历史日期，长途模式的腰部支撑当日是多少？",
      "只依据不晚于10月16日的记忆回答，那个截面上的长途腰托有效值是什么？",
      "回到10月16日当天结束的历史版本，长途驾驶腰撑有效档数是多少？",
    ];
    for (const query of cutoffQueries) {
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query, {
        requestTime: "2026-10-20T01:00:00.000Z",
        timezone: "Asia/Shanghai",
      }));
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
      expect(plan, query).toMatchObject({ highRisk: true, cutoffDate: "2026-10-16" });
    }

    const singerQuery = "有证据能确认陆青喜欢哪位主唱吗？不能由曲风反推出人名。";
    const singerPrepared = prepareCockpitQuery(singerQuery);
    const singerPlan = buildCockpitRetrievalPlan(singerQuery, singerPrepared);
    expect(singerPrepared).toMatchObject({ shouldSearchMemory: true, shouldInject: true });
    expect(singerPrepared.reasons).toContain("subject-attribute-query");
    expect(singerPlan).toMatchObject({ highRisk: true, searchL0: true });
  });

  it("compiles colloquial Chinese media, cutoff, meeting, and seat-role surfaces", () => {
    expect(extractChinesePersonTargets(
      "综合不同会话，司机江屿、副驾驶乘员豆豆、后排乘员梁叔分别习惯几度？",
    )).toEqual([
      { name: "江屿", role: "driver", roleLabel: "司机" },
      { name: "豆豆", role: "front-passenger", roleLabel: "副驾驶乘员" },
      { name: "梁叔", role: "rear-passenger", roleLabel: "后排乘员" },
    ]);

    const cutoffQuery = "查询不超过10月16日的事件，按该日末态给出长途座椅支撑档位。";
    const cutoff = compileChineseCockpitSemantics(cutoffQuery);
    const cutoffPlan = buildCockpitRetrievalPlan(cutoffQuery, prepareCockpitQuery(cutoffQuery, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    }));
    expect(cutoff.intents).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
    expect(cutoffPlan).toMatchObject({ highRisk: true, cutoffDate: "2026-10-16" });

    const meetingQuery = "按日还原碰面处，10月15日的值和10月19日的值分别是哪一地点？";
    const meeting = compileChineseCockpitSemantics(meetingQuery);
    const meetingPlan = buildCockpitRetrievalPlan(meetingQuery, prepareCockpitQuery(meetingQuery, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    }));
    expect(meeting).toMatchObject({ domain: "meeting-point" });
    expect(meeting.targets.map((item) => item.key)).toEqual(["meeting-point"]);
    expect(meeting.intents).toContain("two-date-state");
    expect(meetingPlan).toMatchObject({ highRisk: true });

    for (const query of [
      "夜间音频按现行配置回答，播放类型、最高格数和旧类型状态分别是什么？",
      "夜间音响末次修改落成什么样，听哪类、开到几格为止、以前那类还算数吗？",
    ]) {
      const media = compileChineseCockpitSemantics(query);
      expect(media, query).toMatchObject({ domain: "media" });
      expect(media.targets.map((item) => item.key), query).toEqual(["music-type", "volume-limit"]);
      expect(media.intents, query).toEqual(expect.arrayContaining([
        "latest-state", "multi-target-state", "correction-state",
      ]));
      expect(extractCockpitNamedTargets(query), query).toEqual(["音乐类型", "音量上限"]);
    }
  });

  it("compiles productive Chinese boundary, meeting, alias, occupant, and event wording", () => {
    const aliasQuery = "导航里三处常用地点现在各是哪儿：停车老位置、亲属家、门诊处？";
    const alias = compileChineseCockpitSemantics(aliasQuery);
    expect(alias).toMatchObject({
      domain: "navigation-alias",
      intents: expect.arrayContaining(["latest-state", "multi-target-state"]),
    });
    expect(alias.targets.map((item) => item.key)).toEqual([
      "parking-alias", "relative-home-alias", "clinic-alias",
    ]);
    expect(extractCockpitNamedTargets(aliasQuery)).toEqual(["固定车位", "亲友住处", "诊所"]);

    const peopleQueries = [
      {
        query: "司机林澈，右前座乘客豆豆，后座乘客杜姐，分别常用几度空调？",
        people: [
          { name: "林澈", role: "driver", roleLabel: "司机" },
          { name: "豆豆", role: "front-passenger", roleLabel: "右前座乘客" },
          { name: "杜姐", role: "rear-passenger", roleLabel: "后座乘客" },
        ],
      },
      {
        query: "本趟陆青掌舵，可可在旁边，孟姨在后面，请按人匹配常设温度。",
        people: [
          { name: "陆青", role: "driver", roleLabel: "主驾" },
          { name: "可可", role: "front-passenger", roleLabel: "副驾" },
          { name: "孟姨", role: "rear-passenger", roleLabel: "后排" },
        ],
      },
      {
        query: "车内三席是司机许岚、副驾驶乘客小满、后座乘员孟姨，逐一给温度。",
        people: [
          { name: "许岚", role: "driver", roleLabel: "司机" },
          { name: "小满", role: "front-passenger", roleLabel: "副驾驶乘客" },
          { name: "孟姨", role: "rear-passenger", roleLabel: "后座乘员" },
        ],
      },
    ];
    for (const { query, people } of peopleQueries) {
      expect(extractChinesePersonTargets(query), query).toEqual(people);
    }

    for (const query of [
      "见面地点按日期回答，10月15日是哪处，10月19日又是哪处？",
      "对照10月15日、10月19日的见面处，各自有效值是什么？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics, query).toMatchObject({ domain: "meeting-point" });
      expect(semantics.targets.map((item) => item.key), query).toEqual(["meeting-point"]);
      expect(semantics.intents, query).toContain("two-date-state");
    }

    for (const query of [
      "查询范围封顶在10月16日日终，后面的修改不算，长途腰托取哪档？",
      "仅纳入早于或等于10月16日的历史，回答当日末态腰部支撑设置。",
      "把记忆时间窗关在10月16日当天，按边界内事件给出腰托有效值。",
      "历史读取上限就是10月16日，请报这个截点的长途腰托配置。",
      "对10月16日之后的事件一律不采纳，那天收尾时长途腰撑是多少？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query, {
        requestTime: "2026-10-20T01:00:00.000Z",
        timezone: "Asia/Shanghai",
      }));
      expect(semantics.people, query).toEqual([]);
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
      expect(plan, query).toMatchObject({ highRisk: true, cutoffDate: "2026-10-16" });
    }

    const aggregationQuery = "把五趟真正充过的行程按场站并账，最常去哪里，共去了几趟？";
    const aggregation = compileChineseCockpitSemantics(aggregationQuery);
    const aggregationPlan = buildCockpitRetrievalPlan(aggregationQuery, prepareCockpitQuery(aggregationQuery));
    expect(aggregation).toMatchObject({ domain: "charging-event" });
    expect(aggregation.intents).toContain("event-frequency");
    expect(aggregationPlan).toMatchObject({ highRisk: true, searchL0: true });
  });

  it("compiles formal and spoken Chinese cockpit slot grammar without answer values", () => {
    const seatCases = [
      {
        query: "驾驶员唐川、前座右侧乘员小叶、后排乘客周姨分别偏好几度？",
        people: [
          { name: "唐川", role: "driver", roleLabel: "驾驶员" },
          { name: "小叶", role: "front-passenger", roleLabel: "前座右侧乘员" },
          { name: "周姨", role: "rear-passenger", roleLabel: "后排乘客" },
        ],
      },
      {
        query: "程野坐主驾驶，安安坐副驾驶，何叔坐后座，按人报空调温度。",
        people: [
          { name: "程野", role: "driver", roleLabel: "主驾驶" },
          { name: "安安", role: "front-passenger", roleLabel: "副驾" },
          { name: "何叔", role: "rear-passenger", roleLabel: "后排" },
        ],
      },
      {
        query: "车内人员为主驾许禾、前排乘员宁宁、后排乘员孟姨，各自温度偏好？",
        people: [
          { name: "许禾", role: "driver", roleLabel: "主驾" },
          { name: "宁宁", role: "front-passenger", roleLabel: "前排乘员" },
          { name: "孟姨", role: "rear-passenger", roleLabel: "后排乘员" },
        ],
      },
    ];
    for (const { query, people } of seatCases) {
      expect(extractChinesePersonTargets(query), query).toEqual(people);
    }

    const meeting = compileChineseCockpitSemantics(
      "请分别查询10月11日和10月18日当日的会晤地点。",
    );
    expect(meeting).toMatchObject({ domain: "meeting-point" });
    expect(meeting.targets.map((item) => item.key)).toEqual(["meeting-point"]);
    expect(meeting.intents).toContain("two-date-state");

    for (const query of [
      "通勤现行规则里，路线和两种特殊路分别怎么走？",
      "最终通勤策略的择路、高架、收费路三个字段分别是什么？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.targets.map((item) => item.key), query).toEqual([
        "route", "elevated-road", "toll-road",
      ]);
      expect(extractCockpitNamedTargets(query), query).toHaveLength(3);
    }

    for (const query of [
      "夜驾听歌改到最后，报新类型、音量最高档和原类型是否还有效。",
      "夜间音响有效设置包括内容类型、最高格位和被换内容状态，请说完整。",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics, query).toMatchObject({ domain: "media" });
      expect(semantics.targets.map((item) => item.key), query).toEqual([
        "music-type", "volume-limit",
      ]);
      expect(semantics.intents, query).toEqual(expect.arrayContaining([
        "latest-state", "multi-target-state", "correction-state",
      ]));
      expect(extractCockpitNamedTargets(query), query).toEqual(["音乐类型", "音量上限"]);
    }

    const priorityQuery = "电池表来到7%，急充时谁排第一，原来的两项还在前列吗？";
    const priority = compileChineseCockpitSemantics(priorityQuery);
    const priorityPlan = buildCockpitRetrievalPlan(priorityQuery, prepareCockpitQuery(priorityQuery));
    expect(priority).toMatchObject({ domain: "charging-priority" });
    expect(priority.intents).toContain("conditional-priority");
    expect(priorityPlan).toMatchObject({ highRisk: true, searchL0: true });

    for (const query of [
      "历史日期上界为10月12日，按边界还原长途腰托。",
      "只收录10月12日及以前的事件，回答腰撑末态。",
      "记忆最远看到10月12日日终，后面的更新都屏蔽，腰托几档？",
      "只允许日期小于等于10月12日的记录参与，腰撑末态为几档？",
      "把10月12日设成记忆窗口的右边界，腰托当日生效在哪档？",
      "晚于10月12日的一概过滤掉，按余下历史回答腰撑档位。",
      "10月12日是允许查看的最晚一天，请报当日日终腰部支撑。",
      "可以读取的最后日期为10月12日，查询该日结束时的腰托配置。",
      "可见历史的最晚一日定在10月12日，按当天收尾状态回答腰撑档位。",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query, {
        requestTime: "2026-10-20T01:00:00.000Z",
        timezone: "Asia/Shanghai",
      }));
      expect(semantics.people, query).toEqual([]);
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
      expect(plan, query).toMatchObject({ highRisk: true, cutoffDate: "2026-10-12" });
    }

    const alias = compileChineseCockpitSemantics(
      "导航里的固定车位、亲戚家、门诊地点三个称呼当前分别指哪里？",
    );
    expect(alias.targets.map((item) => item.key)).toEqual([
      "parking-alias", "relative-home-alias", "clinic-alias",
    ]);
  });

  it("compiles productive Chinese cockpit paraphrase families into complete contracts", () => {
    for (const query of [
      "五回确实充上电的流水按地点归堆，哪里占头，共几回？",
      "从五宗实际补过能的明细中找众数站，并给它的计数。",
      "五条已完成补给电量的记录按场地汇总，榜首地点和票数为何？",
      "五回电已经补完，按发生地点做频数表，第一名及次数是什么？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query));
      expect(semantics, query).toMatchObject({ domain: "charging-event" });
      expect(semantics.intents, query).toContain("event-frequency");
      expect(plan, query).toMatchObject({ highRisk: true, searchL0: true });
    }

    for (const query of [
      "现行通勤整套怎么走，路径原则、高架段和收费段分别怎样？",
      "现在通勤走法怎么定，普通路线原则和两种例外道路分别是什么规则？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics, query).toMatchObject({ domain: "commute" });
      expect(semantics.targets.map((item) => item.key), query).toEqual([
        "route", "elevated-road", "toll-road",
      ]);
      expect(extractCockpitNamedTargets(query), query).toHaveLength(3);
    }

    for (const noun of ["会师地点", "约碰地址", "会客地点", "接头地点"]) {
      const query = `10月15日与10月19日的${noun}分别在哪里？`;
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics, query).toMatchObject({ domain: "meeting-point" });
      expect(semantics.targets.map((item) => item.key), query).toEqual(["meeting-point"]);
      expect(semantics.dateMentions, query).toHaveLength(2);
      expect(semantics.intents, query).toContain("two-date-state");
    }

    const peopleCases = [
      {
        query: "林澈开车，豆豆坐前座右手边，杜姐坐第二排，分别喜欢多少度？",
        names: ["林澈", "豆豆", "杜姐"],
      },
      {
        query: "驾驶席上沈舟、邻座宁宁、后舱梁叔三人的空调偏好各是多少？",
        names: ["沈舟", "宁宁", "梁叔"],
      },
      {
        query: "本车周遥负责驾驶，小满在前座右侧，魏叔在后座，逐人报温度。",
        names: ["周遥", "小满", "魏叔"],
      },
      {
        query: "温言坐驾驶位，阿禾在其右侧前座，杜姐在后一排，三位温度各多少？",
        names: ["温言", "阿禾", "杜姐"],
      },
      {
        query: "这一趟江屿握方向盘，豆豆坐前排右手，梁叔坐车后部，请按人给温度。",
        names: ["江屿", "豆豆", "梁叔"],
      },
      {
        query: "许岚担任驾驶者，小满是前舱右座乘客，孟姨是后舱乘客，各偏好多少度？",
        names: ["许岚", "小满", "孟姨"],
      },
      {
        query: "主驾乘员陆青，右前方乘员可可，后排乘员孟姨，分别设几度？",
        names: ["陆青", "可可", "孟姨"],
      },
      {
        query: "主驾驶温言、右前乘员阿禾、后座乘员杜姐同时出行，各用什么温度？",
        names: ["温言", "阿禾", "杜姐"],
      },
      {
        query: "请逐人核对空调记忆：主驾驶江屿，右前座豆豆，后排乘客梁叔。",
        names: ["江屿", "豆豆", "梁叔"],
      },
      {
        query: "主驾驶乘员顾原、副驾驶乘员阿禾、后排乘员孔老师各习惯多少度？",
        names: ["顾原", "阿禾", "孔老师"],
      },
      {
        query: "综合不同对话，主驾驶乘员江屿、副驾驶乘客豆豆、后座乘员梁叔各几度？",
        names: ["江屿", "豆豆", "梁叔"],
      },
      {
        query: "驾驶位上的林澈、前座右侧的豆豆、后舱里的杜姐分别习惯几度？",
        names: ["林澈", "豆豆", "杜姐"],
      },
    ];
    for (const { query, names } of peopleCases) {
      const people = extractChinesePersonTargets(query);
      expect(people.map((item) => item.name), query).toEqual(names);
      expect(people.map((item) => item.role), query).toEqual([
        "driver", "front-passenger", "rear-passenger",
      ]);
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics, query).toMatchObject({ domain: "occupant-temperature" });
      expect(semantics.intents, query).toContain("multi-person-state");
    }

    for (const query of [
      "历史资料只算到10月16日为止，按含当天的截面回答长途腰托档位。",
      "事件日期不得大于10月16日，在这个闭区间内还原长途腰部支撑。",
      "最后可见历史日定为10月16日，请给边界日收尾的长途腰撑值。",
      "将记忆时间轴裁到10月16日日终，按裁剪后的末态回答腰托设置。",
      "查询窗口的右端是10月16日，只用端点以内事件回答腰托档数。",
    ]) {
      const prepared = prepareCockpitQuery(query, {
        requestTime: "2026-10-20T01:00:00.000Z",
        timezone: "Asia/Shanghai",
      });
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepared);
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
      expect(plan, query).toMatchObject({ highRisk: true, cutoffDate: "2026-10-16" });
    }

    const priorityQuery = "电池读数只有10%，临时找桩先比哪项，休息室与口碑怎么排？";
    const priority = compileChineseCockpitSemantics(priorityQuery);
    expect(priority, priorityQuery).toMatchObject({ domain: "charging-priority" });
    expect(priority.intents, priorityQuery).toContain("conditional-priority");
    expect(buildCockpitRetrievalPlan(priorityQuery, prepareCockpitQuery(priorityQuery))).toMatchObject({
      highRisk: true,
      searchL0: true,
    });

    const missingSingerQuery = "若历史仅有曲风，能否知道某位乘员喜欢谁唱？请据实拒答。";
    const missingSingerPrepared = prepareCockpitQuery(missingSingerQuery);
    const missingSingerPlan = buildCockpitRetrievalPlan(missingSingerQuery, missingSingerPrepared);
    expect(missingSingerPrepared.shouldSearchMemory).toBe(true);
    expect(missingSingerPrepared.reasons).toContain("subject-attribute-query");
    expect(missingSingerPlan).toMatchObject({ highRisk: true });
    expect(missingSingerPlan.risks).toContain("subject-attribute-query");

    const aliasQuery = "核对常用地点绑定：停车惯用处、亲友住所、诊疗地点各是什么？";
    expect(compileChineseCockpitSemantics(aliasQuery).targets.map((item) => item.key)).toEqual([
      "parking-alias", "relative-home-alias", "clinic-alias",
    ]);
    expect(extractCockpitNamedTargets(aliasQuery)).toHaveLength(3);

    const mediaQuery = "夜驾声音方案收口后，播的是什么、声量上限几格、先前内容还生效吗？";
    const media = compileChineseCockpitSemantics(mediaQuery);
    expect(media, mediaQuery).toMatchObject({ domain: "media" });
    expect(media.targets.map((item) => item.key)).toEqual(["music-type", "volume-limit"]);
    expect(media.intents, mediaQuery).toEqual(expect.arrayContaining([
      "latest-state", "multi-target-state", "correction-state",
    ]));
  });

  it("compiles productive Chinese frequency, low-energy ranking, and cutoff declarations", () => {
    for (const query of [
      "把五次确已结束的补电按场地合账，出现最勤的是哪处，一共几次？",
      "盘一盘五次真正补过电的行程，最常到访哪站，总计几回？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query));
      expect(semantics, query).toMatchObject({ domain: "charging-event" });
      expect(semantics.intents, query).toContain("event-frequency");
      expect(plan, query).toMatchObject({ highRisk: true, searchL0: true });
    }

    const priorityQuery = "表显降至8%就去找桩，远近会排第几，休息室和口碑如何安排？";
    const priority = compileChineseCockpitSemantics(priorityQuery);
    const priorityPlan = buildCockpitRetrievalPlan(priorityQuery, prepareCockpitQuery(priorityQuery));
    expect(priority, priorityQuery).toMatchObject({ domain: "charging-priority" });
    expect(priority.intents, priorityQuery).toContain("conditional-priority");
    expect(priorityPlan, priorityQuery).toMatchObject({ highRisk: true, searchL0: true });

    const cutoffQuery = "以10月16日作为记忆查询上界，忽略更晚事件，腰部支撑末值是什么？";
    const cutoff = compileChineseCockpitSemantics(cutoffQuery);
    const cutoffPlan = buildCockpitRetrievalPlan(cutoffQuery, prepareCockpitQuery(cutoffQuery, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    }));
    expect(cutoff.intents, cutoffQuery).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
    expect(cutoffPlan, cutoffQuery).toMatchObject({ highRisk: true, cutoffDate: "2026-10-16" });
  });

  it("compiles elliptical Chinese field, lifecycle, priority, and cutoff contracts", () => {
    const employeeCard = prepareCockpitQuery("记忆是否明确保存某人员工卡ID，没有精确值就拒答。");
    const employeePlan = buildCockpitRetrievalPlan("记忆是否明确保存某人员工卡ID，没有精确值就拒答。", employeeCard);
    expect(employeeCard.shouldSearchMemory).toBe(true);
    expect(employeePlan).toMatchObject({ highRisk: true, searchL0: true });

    const mediaQuery = "车内晚间音频现值请答全：新门类、音量封顶、老门类去留。";
    const media = compileChineseCockpitSemantics(mediaQuery);
    const mediaPlan = buildCockpitRetrievalPlan(mediaQuery, prepareCockpitQuery(mediaQuery));
    expect(media.targets.map((item) => item.key)).toEqual(["music-type", "volume-limit"]);
    expect(media.intents).toEqual(expect.arrayContaining(["latest-state", "correction-state", "multi-target-state"]));
    expect(mediaPlan.highRisk).toBe(true);

    const cancellationQuery = "车检排期收尾的现存项、最新撤项、随后追加项分别是什么？";
    const cancellation = compileChineseCockpitSemantics(cancellationQuery);
    const cancellationPlan = buildCockpitRetrievalPlan(cancellationQuery, prepareCockpitQuery(cancellationQuery));
    expect(cancellation.intents).toEqual(expect.arrayContaining(["latest-state", "final-cancellation"]));
    expect(extractCockpitNamedTargets(cancellationQuery)).toEqual([]);
    expect(cancellationPlan.highRisk).toBe(true);

    const priorityQuery = "仪表显示10%的低电量下，三个条件谁置首，其他怎么排？";
    const priority = compileChineseCockpitSemantics(priorityQuery);
    expect(priority).toMatchObject({ domain: "charging-priority" });
    expect(priority.intents).toContain("conditional-priority");

    const cutoffQuery = "按10月16日及更早历史还原长途支撑，不能采用后来的修改。";
    const cutoff = compileChineseCockpitSemantics(cutoffQuery);
    const cutoffPlan = buildCockpitRetrievalPlan(cutoffQuery, prepareCockpitQuery(cutoffQuery, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    }));
    expect(cutoff.intents).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
    expect(cutoffPlan).toMatchObject({ highRisk: true, cutoffDate: "2026-10-16" });
  });

  it("compiles the revealed RC31 Chinese grammar families into complete recall contracts", () => {
    for (const query of [
      "仪表只显示10%准备找充电站，距离、休息室、评分的先后怎样？",
      "低电量为10%时立刻选桩，最近距离和另外两项各排在什么位置？",
      "仪表电量仅余10%，三个选址依据的首要项和剩余项分别怎样？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const prepared = prepareCockpitQuery(query);
      const plan = buildCockpitRetrievalPlan(query, prepared);
      expect(semantics, query).toMatchObject({ domain: "charging-priority" });
      expect(semantics.intents, query).toContain("conditional-priority");
      expect(prepared.shouldSearchMemory, query).toBe(true);
      expect(plan, query).toMatchObject({ highRisk: true, searchL0: true });
    }

    const mediaQuery = "车内夜听配置的最终版本是什么，内容类型、最高档位、被替换内容都要答。";
    const media = compileChineseCockpitSemantics(mediaQuery);
    expect(media, mediaQuery).toMatchObject({ domain: "media" });
    expect(media.targets.map((item) => item.key), mediaQuery).toEqual(["music-type", "volume-limit"]);
    expect(media.intents, mediaQuery).toEqual(expect.arrayContaining([
      "latest-state", "multi-target-state", "correction-state",
    ]));
    expect(buildCockpitRetrievalPlan(mediaQuery, prepareCockpitQuery(mediaQuery)), mediaQuery)
      .toMatchObject({ highRisk: true, searchL0: true });

    const meetingQuery = "10月15日当日约在哪儿见，到了10月19日当日又约在哪儿见？";
    const meeting = compileChineseCockpitSemantics(meetingQuery);
    expect(meeting, meetingQuery).toMatchObject({ domain: "meeting-point" });
    expect(meeting.targets.map((item) => item.key), meetingQuery).toEqual(["meeting-point"]);
    expect(meeting.dateMentions, meetingQuery).toHaveLength(2);
    expect(buildCockpitRetrievalPlan(meetingQuery, prepareCockpitQuery(meetingQuery)), meetingQuery)
      .toMatchObject({ highRisk: true, searchL0: true });

    const absentCredentialQuery = "返回周遥单位出入凭证的编号；若没有直接字段就答无法确定。";
    const absentCredential = prepareCockpitQuery(absentCredentialQuery);
    const absentCredentialPlan = buildCockpitRetrievalPlan(absentCredentialQuery, absentCredential);
    expect(absentCredential.shouldSearchMemory, absentCredentialQuery).toBe(true);
    expect(absentCredential.reasons, absentCredentialQuery).toContain("subject-attribute-query");
    expect(absentCredentialPlan, absentCredentialQuery).toMatchObject({ highRisk: true, searchL0: true });

    for (const query of [
      "查询窗口包含10月16日但不包含更晚日期，窗口末端的腰托配置是什么？",
      "时间线只看到10月16日为止，请回答那一天收尾时生效的腰部支撑档位。",
      "以10月16日作为可见记录的终点，边界后的信息不得用于回答腰托状态。",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query, {
        requestTime: "2026-10-20T01:00:00.000Z",
        timezone: "Asia/Shanghai",
      }));
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
      expect(plan, query).toMatchObject({ highRisk: true, searchL0: true, cutoffDate: "2026-10-16" });
    }

    for (const query of [
      "核对车辆年审终态：当前有效预约、末次删除记录、删除后安排分别为何。",
      "车辆检验排期最终版里保留、撤销、撤销后新建三类项目各怎样？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query));
      expect(semantics, query).toMatchObject({ domain: "inspection" });
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "final-cancellation"]));
      expect(extractCockpitNamedTargets(query), query).toEqual([]);
      expect(plan, query).toMatchObject({ highRisk: true, searchL0: true });
    }

    const commuteQuery = "导航多次修改后最终执行什么通勤口径，三项道路规则逐一说明。";
    const commute = compileChineseCockpitSemantics(commuteQuery);
    expect(commute, commuteQuery).toMatchObject({ domain: "commute" });
    expect(commute.targets.map((item) => item.key), commuteQuery).toEqual([
      "route", "elevated-road", "toll-road",
    ]);
    expect(buildCockpitRetrievalPlan(commuteQuery, prepareCockpitQuery(commuteQuery)), commuteQuery)
      .toMatchObject({ highRisk: true, searchL0: true });

    const peopleQuery = "车内由苏木驾驶，宁宁坐其右侧，魏叔坐后方，请按人给空调温度。";
    const people = extractChinesePersonTargets(peopleQuery);
    expect(people, peopleQuery).toEqual([
      expect.objectContaining({ name: "苏木", role: "driver" }),
      expect.objectContaining({ name: "宁宁", role: "front-passenger" }),
      expect.objectContaining({ name: "魏叔", role: "rear-passenger" }),
    ]);
    expect(compileChineseCockpitSemantics(peopleQuery).intents, peopleQuery).toContain("multi-person-state");
    expect(buildCockpitRetrievalPlan(peopleQuery, prepareCockpitQuery(peopleQuery)), peopleQuery)
      .toMatchObject({ highRisk: true, searchL0: true });
  });

  it("compiles the revealed RC32 Chinese grammar families without inventing an unspecified field", () => {
    const cutoffQuery = "本次查询纳入10月16日但屏蔽更晚事件，请给窗口末端的腰部支撑。";
    const cutoff = compileChineseCockpitSemantics(cutoffQuery);
    const cutoffPlan = buildCockpitRetrievalPlan(cutoffQuery, prepareCockpitQuery(cutoffQuery, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    }));
    expect(cutoff, cutoffQuery).toMatchObject({ domain: "lumbar" });
    expect(cutoff.intents, cutoffQuery).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
    expect(cutoffPlan, cutoffQuery).toMatchObject({ highRisk: true, searchL0: true, cutoffDate: "2026-10-16" });

    const meetingQuery = "10月15日约见应去何处，10月19日约见又应去何处？";
    const meeting = compileChineseCockpitSemantics(meetingQuery);
    expect(meeting, meetingQuery).toMatchObject({ domain: "meeting-point" });
    expect(meeting.targets.map((item) => item.key), meetingQuery).toEqual(["meeting-point"]);
    expect(meeting.intents, meetingQuery).toContain("two-date-state");
    expect(buildCockpitRetrievalPlan(meetingQuery, prepareCockpitQuery(meetingQuery)), meetingQuery)
      .toMatchObject({ highRisk: true, searchL0: true });

    const farTripSupportQuery = "记录时间线截至10月16日就停止，在该截点有效的远途支撑是多少？";
    const farTripSupport = compileChineseCockpitSemantics(farTripSupportQuery);
    expect(farTripSupport, farTripSupportQuery).toMatchObject({ domain: "lumbar" });
    expect(farTripSupport.intents, farTripSupportQuery)
      .toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));

    const priorityQuery = "电池读数来到8%必须补电，三类选址因素谁先谁后？";
    const priority = compileChineseCockpitSemantics(priorityQuery);
    expect(priority, priorityQuery).toMatchObject({ domain: "charging-priority" });
    expect(priority.intents, priorityQuery).toContain("conditional-priority");
    expect(buildCockpitRetrievalPlan(priorityQuery, prepareCockpitQuery(priorityQuery)), priorityQuery)
      .toMatchObject({ highRisk: true, searchL0: true });

    const singerQuery = "返回温言喜欢的歌手人名，不允许根据曲风反向猜人。";
    const singerPrepared = prepareCockpitQuery(singerQuery);
    expect(singerPrepared, singerQuery).toMatchObject({ shouldSearchMemory: true, shouldInject: true });
    expect(singerPrepared.reasons, singerQuery).toContain("subject-attribute-query");
    expect(buildCockpitRetrievalPlan(singerQuery, singerPrepared), singerQuery)
      .toMatchObject({ highRisk: true, searchL0: true });

    const aliasQuery = "三个地点简称的现值是什么：常停点、家人住所、挂号地点？";
    const aliases = compileChineseCockpitSemantics(aliasQuery);
    expect(aliases, aliasQuery).toMatchObject({ domain: "navigation-alias" });
    expect(aliases.targets.map((item) => item.key), aliasQuery).toEqual([
      "parking-alias", "relative-home-alias", "clinic-alias",
    ]);
    expect(extractCockpitNamedTargets(aliasQuery), aliasQuery).toEqual(["固定车位", "亲友住处", "诊所"]);

    const ambiguousQuery = "以10月16日作为历史终点还原长途配置，边界后的修改全部忽略。";
    const ambiguous = compileChineseCockpitSemantics(ambiguousQuery);
    expect(ambiguous.intents, ambiguousQuery).toContain("cutoff-state");
    expect(ambiguous.domain, ambiguousQuery).toBe("unknown");
    expect(ambiguous.targets, ambiguousQuery).toEqual([]);
  });

  it("compiles the revealed RC33 battery, cutoff, and umbrella-slot grammar", () => {
    const priorityQuery = "仪表读数降到10%要马上充能，三个选址标准按怎样的先后执行？";
    const priority = compileChineseCockpitSemantics(priorityQuery);
    const priorityPlan = buildCockpitRetrievalPlan(priorityQuery, prepareCockpitQuery(priorityQuery));
    expect(priority, priorityQuery).toMatchObject({ domain: "charging-priority" });
    expect(priority.intents, priorityQuery).toContain("conditional-priority");
    expect(priorityPlan, priorityQuery).toMatchObject({ highRisk: true, searchL0: true });

    const cutoffQuery = "查询时间线只推进至10月16日，请回答截点上的远途支撑档数。";
    const cutoff = compileChineseCockpitSemantics(cutoffQuery);
    const cutoffPlan = buildCockpitRetrievalPlan(cutoffQuery, prepareCockpitQuery(cutoffQuery, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    }));
    expect(cutoff, cutoffQuery).toMatchObject({ domain: "lumbar" });
    expect(cutoff.intents, cutoffQuery).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
    expect(cutoffPlan, cutoffQuery).toMatchObject({ highRisk: true, searchL0: true, cutoffDate: "2026-10-16" });

    const commuteQuery = "通勤选路更新收尾后，普通路线依据和两类道路处置分别是什么？";
    const commute = compileChineseCockpitSemantics(commuteQuery);
    expect(commute, commuteQuery).toMatchObject({ domain: "commute" });
    expect(commute.targets.map((item) => item.key), commuteQuery).toEqual([
      "route", "elevated-road", "toll-road",
    ]);
    expect(extractCockpitNamedTargets(commuteQuery), commuteQuery).toEqual([
      "选路", "高架", "收费道路",
    ]);
  });

  it("compiles the revealed RC34 productive Chinese grammar and keeps unnamed dates fail-closed", () => {
    for (const query of [
      "车内安排周遥主驾、小满前排右座、魏叔第二排，分别偏好几度？",
      "座舱安排甲主驾驶席、乙前排右座、丙第二排，各自空调要几度？",
      "驾驶席里的周遥、邻座的小满、第二排里的魏叔请逐人查温度。",
      "驾驶座上的苏木、右前方的宁宁、后舱里的魏叔逐一查询常用温度。",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      const plan = buildCockpitRetrievalPlan(query, prepareCockpitQuery(query));
      expect(semantics, query).toMatchObject({ domain: "occupant-temperature" });
      expect(semantics.people.map((person) => person.role), query).toEqual([
        "driver", "front-passenger", "rear-passenger",
      ]);
      expect(semantics.intents, query).toContain("multi-person-state");
      expect(plan, query).toMatchObject({ highRisk: true, searchL0: true });
    }

    const ellipticalPeopleQuery = "主驾驶席是程野，副驾驶是小葵，后排是孔老师，三位设置各为何？";
    const ellipticalPeople = compileChineseCockpitSemantics(ellipticalPeopleQuery);
    expect(ellipticalPeople, ellipticalPeopleQuery).toMatchObject({ domain: "occupant-temperature" });
    expect(ellipticalPeople.people.map((person) => person.name), ellipticalPeopleQuery)
      .toEqual(["程野", "小葵", "孔老师"]);
    expect(ellipticalPeople.intents, ellipticalPeopleQuery).toContain("multi-person-state");
    expect(buildCockpitRetrievalPlan(ellipticalPeopleQuery, prepareCockpitQuery(ellipticalPeopleQuery)))
      .toMatchObject({ highRisk: true, searchL0: true });

    const cutoffQuery = "把历史裁剪到10月16日当天结束，求裁剪后最后有效的腰托档数。";
    const cutoff = compileChineseCockpitSemantics(cutoffQuery);
    expect(cutoff, cutoffQuery).toMatchObject({ domain: "lumbar" });
    expect(cutoff.intents, cutoffQuery).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
    expect(buildCockpitRetrievalPlan(cutoffQuery, prepareCockpitQuery(cutoffQuery, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    })), cutoffQuery).toMatchObject({ highRisk: true, searchL0: true, cutoffDate: "2026-10-16" });

    const priorityQuery = "余量降为8%后选补能点，最近距离和另外两项各是什么顺位？";
    const priority = compileChineseCockpitSemantics(priorityQuery);
    expect(priority, priorityQuery).toMatchObject({ domain: "charging-priority" });
    expect(priority.intents, priorityQuery).toContain("conditional-priority");
    expect(buildCockpitRetrievalPlan(priorityQuery, prepareCockpitQuery(priorityQuery)), priorityQuery)
      .toMatchObject({ highRisk: true, searchL0: true });

    const missingDateIdentityQuery = "查询双时点会客地点，两个日期的答案不能合并。";
    const missingDateIdentity = compileChineseCockpitSemantics(missingDateIdentityQuery);
    const missingDatePlan = buildCockpitRetrievalPlan(
      missingDateIdentityQuery,
      prepareCockpitQuery(missingDateIdentityQuery),
    );
    expect(missingDateIdentity.intents, missingDateIdentityQuery).toContain("two-date-state");
    expect(missingDatePlan, missingDateIdentityQuery).toMatchObject({
      highRisk: true,
      searchL0: true,
      requiredDates: [],
    });
  });

  it("compiles RC35 shared-predicate dates, elliptical media fields, and long seat suffixes", () => {
    const peopleQuery = "本趟乘员安排为林澈主驾驶位，豆豆副驾驶席，杜姐后排座，各人的温度设置是什么？";
    const people = compileChineseCockpitSemantics(peopleQuery);
    expect(people, peopleQuery).toMatchObject({ domain: "occupant-temperature" });
    expect(people.people.map((person) => [person.name, person.role]), peopleQuery).toEqual([
      ["林澈", "driver"],
      ["豆豆", "front-passenger"],
      ["杜姐", "rear-passenger"],
    ]);
    expect(buildCockpitRetrievalPlan(peopleQuery, prepareCockpitQuery(peopleQuery)), peopleQuery)
      .toMatchObject({ highRisk: true, searchL0: true });

    const mediaQuery = "车机夜听现行配置要答齐，当前内容、音量限制及先前内容状态分别为何？";
    const media = compileChineseCockpitSemantics(mediaQuery);
    expect(media, mediaQuery).toMatchObject({ domain: "media" });
    expect(media.targets.map((target) => target.key), mediaQuery).toEqual([
      "music-type", "volume-limit",
    ]);
    expect(extractCockpitNamedTargets(mediaQuery), mediaQuery).toEqual([
      "音乐类型", "音量上限",
    ]);

    const twoDateQuery = "10月15日应去什么位置会面，到了10月19日应去哪里？";
    const twoDate = compileChineseCockpitSemantics(twoDateQuery);
    const twoDatePlan = buildCockpitRetrievalPlan(twoDateQuery, prepareCockpitQuery(twoDateQuery, {
      requestTime: "2026-10-20T01:00:00.000Z",
      timezone: "Asia/Shanghai",
    }));
    expect(twoDate, twoDateQuery).toMatchObject({ domain: "meeting-point" });
    expect(twoDate.targets.map((target) => target.key), twoDateQuery).toEqual(["meeting-point"]);
    expect(twoDate.intents, twoDateQuery).toContain("two-date-state");
    expect(twoDatePlan, twoDateQuery).toMatchObject({
      highRisk: true,
      searchL0: true,
      requiredDates: ["10月15日", "10月19日"],
    });
  });

  it("compiles RC36 Chinese seat, cutoff, meeting, and aggregation precedence forms", () => {
    const peopleCases: Array<[string, Array<[string, string]>]> = [
      [
        "座次表是顾原主驾驶位、阿禾副驾驶位及孔老师后排座位，请逐人查询空调温度。",
        [["顾原", "driver"], ["阿禾", "front-passenger"], ["孔老师", "rear-passenger"]],
      ],
      [
        "温言在驾驶座位、阿禾在前舱右座、杜姐在后舱座位，请核对每人的温度。",
        [["温言", "driver"], ["阿禾", "front-passenger"], ["杜姐", "rear-passenger"]],
      ],
      [
        "本车座位绑定为许岚主驾位、小满前排右位、孟姨后排位，请把温度逐项配对。",
        [["许岚", "driver"], ["小满", "front-passenger"], ["孟姨", "rear-passenger"]],
      ],
      [
        "主驾驶席上的沈舟、前舱右座的宁宁、后排座位的梁叔，请按人查温度。",
        [["沈舟", "driver"], ["宁宁", "front-passenger"], ["梁叔", "rear-passenger"]],
      ],
      [
        "驾驶座位是江屿，副驾位是豆豆，后舱座位是梁叔，请逐人给温度偏好。",
        [["江屿", "driver"], ["豆豆", "front-passenger"], ["梁叔", "rear-passenger"]],
      ],
      [
        "座舱座次表列着程野驾驶席、小葵前排右位和孔老师第二排，请逐个回答温度。",
        [["程野", "driver"], ["小葵", "front-passenger"], ["孔老师", "rear-passenger"]],
      ],
      [
        "三位乘员依次是苏木、宁宁、魏叔，座位对应主驾、副驾、后排，分别几度？",
        [["苏木", "driver"], ["宁宁", "front-passenger"], ["魏叔", "rear-passenger"]],
      ],
      [
        "席位绑定显示许岚主驾位、小满前舱右座位、孟姨后舱座，请逐人回答温度偏好。",
        [["许岚", "driver"], ["小满", "front-passenger"], ["孟姨", "rear-passenger"]],
      ],
    ];
    for (const [query, expectedPeople] of peopleCases) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics, query).toMatchObject({ domain: "occupant-temperature" });
      expect(semantics.people.map((person) => [person.name, person.role]), query).toEqual(expectedPeople);
      expect(semantics.intents, query).toContain("multi-person-state");
      expect(buildCockpitRetrievalPlan(query, prepareCockpitQuery(query)), query)
        .toMatchObject({ highRisk: true, searchL0: true });
    }

    for (const query of [
      "只读取到10月16日当天结束的记录，长途腰托在这个右边界是几档？",
      "不采用晚于10月16日的记录，请查长途腰撑在当日终点的有效档数。",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics, query).toMatchObject({ domain: "lumbar" });
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
      expect(buildCockpitRetrievalPlan(query, prepareCockpitQuery(query, {
        requestTime: "2026-10-20T01:00:00.000Z",
        timezone: "Asia/Shanghai",
      })), query).toMatchObject({ highRisk: true, searchL0: true, cutoffDate: "2026-10-16" });
    }

    const meetingQuery = "同一个约见事项在10月15日、10月19日的有效位置分别是什么？";
    const meeting = compileChineseCockpitSemantics(meetingQuery);
    expect(meeting, meetingQuery).toMatchObject({ domain: "meeting-point" });
    expect(meeting.targets.map((target) => target.key), meetingQuery).toEqual(["meeting-point"]);
    expect(buildCockpitRetrievalPlan(meetingQuery, prepareCockpitQuery(meetingQuery)), meetingQuery)
      .toMatchObject({ highRisk: true, searchL0: true, requiredDates: ["10月15日", "10月19日"] });

    const aggregationQuery = "盘点五次已完成充电，同一站合在一起，最终哪座场站占比最多、有几次？";
    const prepared = prepareCockpitQuery(aggregationQuery);
    expect(prepared.reasons, aggregationQuery).toContain("aggregation-frequency");
    expect(prepared.reasons, aggregationQuery).not.toContain("latest-final-update");
    expect(buildCockpitRetrievalPlan(aggregationQuery, prepared).risks, aggregationQuery)
      .toEqual(["aggregation-frequency"]);
  });

  it("compiles RC37 productive seat introducers, meeting compounds, and alias surfaces", () => {
    const peopleQueries = [
      "座位表显示顾原主驾驶座、阿禾副驾驶座和孔老师后舱座，请逐个报空调温度。",
      "座位绑定写明顾原主驾驶座、阿禾副驾驶座、孔老师后排座位，请按人给出常用空调温度。",
      "座次表为顾原主驾、阿禾前排右席、孔老师后排乘员，请分别回答常用温度。",
    ];
    for (const peopleQuery of peopleQueries) {
      const people = compileChineseCockpitSemantics(peopleQuery);
      expect(people, peopleQuery).toMatchObject({ domain: "occupant-temperature" });
      expect(people.people.map((person) => [person.name, person.role]), peopleQuery).toEqual([
        ["顾原", "driver"],
        ["阿禾", "front-passenger"],
        ["孔老师", "rear-passenger"],
      ]);
      expect(buildCockpitRetrievalPlan(peopleQuery, prepareCockpitQuery(peopleQuery)), peopleQuery)
        .toMatchObject({ highRisk: true, searchL0: true });
    }

    const meetingQuery = "请逐日返回会客落点，先答10月15日，再答10月19日。";
    const meeting = compileChineseCockpitSemantics(meetingQuery);
    expect(meeting, meetingQuery).toMatchObject({ domain: "meeting-point" });
    expect(meeting.targets.map((target) => target.key), meetingQuery).toEqual(["meeting-point"]);
    expect(buildCockpitRetrievalPlan(meetingQuery, prepareCockpitQuery(meetingQuery)), meetingQuery)
      .toMatchObject({
        highRisk: true,
        searchL0: true,
        requiredDates: ["10月15日", "10月19日"],
      });

    const aliasQuery = "逐个解读有效导航简称：固定停车点指哪儿，亲属家与门诊地址又各是哪儿？";
    const aliases = compileChineseCockpitSemantics(aliasQuery);
    const prepared = prepareCockpitQuery(aliasQuery);
    expect(aliases, aliasQuery).toMatchObject({ domain: "navigation-alias" });
    expect(aliases.targets.map((target) => target.key), aliasQuery).toEqual([
      "parking-alias", "relative-home-alias", "clinic-alias",
    ]);
    expect(aliases.intents, aliasQuery).toEqual(expect.arrayContaining(["latest-state", "multi-target-state"]));
    expect(extractCockpitNamedTargets(aliasQuery), aliasQuery).toEqual([
      "固定车位", "亲友住处", "诊所",
    ]);
    expect(prepared.reasons, aliasQuery).toContain("latest-final-update");
    expect(buildCockpitRetrievalPlan(aliasQuery, prepared).risks, aliasQuery)
      .toEqual(["latest-final-update", "subject-attribute-query"]);

    const legacyAliasQuery = "车载记忆中的固定车位在哪里，亲友住处在哪里，诊所又在哪里？";
    const legacyAliases = compileChineseCockpitSemantics(legacyAliasQuery);
    expect(legacyAliases.targets.map((target) => target.key), legacyAliasQuery).toEqual([
      "parking-alias", "relative-home-alias", "clinic-alias",
    ]);
    expect(extractCockpitNamedTargets(legacyAliasQuery), legacyAliasQuery).toEqual([
      "固定车位", "亲友住处", "诊所",
    ]);
    expect(prepareCockpitQuery(legacyAliasQuery).reasons, legacyAliasQuery)
      .toContain("latest-final-update");

    const parkingOnlyQuery = "常停车处目前会导航到哪里？";
    const parkingOnly = compileChineseCockpitSemantics(parkingOnlyQuery);
    expect(parkingOnly, parkingOnlyQuery).toMatchObject({ domain: "navigation-alias" });
    expect(parkingOnly.targets.map((target) => target.key), parkingOnlyQuery).toEqual(["parking-alias"]);
  });

  it("grounds every slot in a Chinese authoritative final snapshot", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "commute-v1", type: "preference", content: "通勤先按红绿灯最少的路线走，高架和收费道路都绕开。", updated_at: "2026-10-03T07:00:00+08:00", score: 0.8 },
        { id: "commute-temp", type: "preference", content: "临时选择最快路线；允许走高架，收费道路仍然避开。", updated_at: "2026-10-07T07:00:00+08:00", score: 0.82 },
        { id: "commute-final", type: "preference", content: "最终通勤选择里程最短路线；高架可以通行；收费道路继续避开。", updated_at: "2026-10-12T07:00:00+08:00", score: 0.9 },
        { id: "later-neighbor", type: "preference", content: "10月18日把车辆导航的会合点恢复为东岭智造园。", updated_at: "2026-10-18T07:00:00+08:00", score: 0.95 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(CHINESE_MULTI_SLOT_FINAL_QUERY)))[0];
    const prompt = block?.content ?? "";

    expect(prompt).toContain('sufficient="true"');
    expect(prompt).toContain('路线选择="最终通勤选择里程最短路线"');
    expect(prompt).toContain('高架通行="高架可以通行"');
    expect(prompt).toContain('收费道路="收费道路继续避开"');
    expect(block?.metadata?.cockpitAnswerContract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { label: "路线选择", value: "最终通勤选择里程最短路线" },
        { label: "高架通行", value: "高架可以通行" },
        { label: "收费道路", value: "收费道路继续避开" },
      ]),
      fallbackAnswer: expect.stringContaining("里程最短路线"),
    });
  });

  it("compiles a natural spoken route list into a complete final-state contract", async () => {
    const query = "通勤方案定下来之后，如今路线选择、高架、收费路三条规则各是什么？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "old", type: "preference", content: "用户于2026年10月7日临时选择最快路线；高架绕行；收费道路避开。", score: 0.8 },
        { id: "final", type: "preference", content: "用户于2026年10月12日最终通勤选择里程最短路线；高架可以通行；收费道路继续避开。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(query)))[0];

    expect(block?.metadata?.triggerReasons).toContain("latest-final-update");
    expect(block?.metadata?.cockpitAnswerContract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { label: "路线选择", value: "最终通勤选择里程最短路线" },
        { label: "高架", value: "高架可以通行" },
        { label: "收费路", value: "收费道路继续避开" },
      ]),
    });
  });

  it("expands an implicit two-road commute question into three grounded slots", async () => {
    const query = "经过多次调整，正在执行的路线原则以及两类道路限制是什么？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "old", type: "preference", content: "用户于2026年10月7日先选择最快路线；高架绕行；收费道路避开。", score: 0.8 },
        { id: "final", type: "preference", content: "用户于2026年10月12日最终通勤选择里程最短路线；高架可以通行；收费道路继续避开。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const contract = (await injector().execute(context(query)))[0]?.metadata?.cockpitAnswerContract;

    expect(contract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { label: "路线原则", value: "最终通勤选择里程最短路线" },
        { label: "高架", value: "高架可以通行" },
        { label: "收费道路", value: "收费道路继续避开" },
      ]),
      fallbackAnswer: expect.stringContaining("里程最短路线"),
    });
  });

  it("grounds an implicit navigation criterion and two road constraints", async () => {
    const query = "终版导航策略包含选路准则和两种道路约束，请把三项完整列出。";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "old", type: "preference", content: "用户于2026年10月7日先选择最快路线；高架绕行；收费道路避开。", score: 0.8 },
        { id: "final", type: "preference", content: "用户于2026年10月12日最终通勤选择里程最短路线；高架可以通行；收费道路继续避开。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const contract = (await injector().execute(context(query)))[0]?.metadata?.cockpitAnswerContract;

    expect(contract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { label: "选路准则", value: "最终通勤选择里程最短路线" },
        { label: "高架", value: "高架可以通行" },
        { label: "收费道路", value: "收费道路继续避开" },
      ]),
      fallbackAnswer: expect.stringContaining("里程最短路线"),
    });
  });

  it("projects arity-checked ordered Chinese alias snapshots across productive target surfaces", async () => {
    const queries = [
      "请列出三个车载别名的现行映射：“固定车位”“亲友住处”“诊所”分别在哪里？",
      "逐个解读有效导航简称：固定停车点指哪儿，亲属家与门诊地址又各是哪儿？",
      "车载记忆中的固定车位在哪里，亲友住处在哪里，诊所又在哪里？",
    ];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "old", type: "alias", content: "用户于2026年10月8日记录了旧的车载别名。", score: 0.8 },
        { id: "final", type: "alias", content: "用户于2026年10月18日统一更新别名：‘固定车位’改成云港商务区；‘亲友住处’仍是听泉路261号；三项当前映射为云港商务区、听泉路261号、安澜诊所1室。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    for (const query of queries) {
      const contract = (await injector().execute(context(query)))[0]?.metadata?.cockpitAnswerContract;

      expect(contract, query).toMatchObject({
        sufficient: true,
        requiredFacts: expect.arrayContaining([
          { label: "固定车位", value: "云港商务区" },
          { label: "亲友住处", value: "听泉路261号" },
          { label: "诊所", value: "安澜诊所1室" },
        ]),
        fallbackAnswer: "固定车位=云港商务区；亲友住处=听泉路261号；诊所=安澜诊所1室",
      });
    }
  });

  it("routes cancellation, revocation, and negation questions through final-state recall", () => {
    const queries = [
      "之前的保养预约已经取消了吗，现在还有效吗？",
      "充电站优先规则撤销以后，当前最终规则是什么？",
      "Is Maya's old inspection still valid, or was it cancelled?",
      "What is the current food policy after dairy is no longer allowed?",
    ];
    for (const query of queries) {
      const prepared = prepareCockpitQuery(query);
      expect(prepared.shouldSearchMemory, query).toBe(true);
      expect(prepared.reasons, query).toContain("latest-final-update");
    }
  });

  it("grounds both the cancelled prior state and final state for correction questions", async () => {
    const query = "机场导航经过改口后，最终确认的机场和航站楼是什么？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "old", type: "episodic", content: "用户先选择虹桥机场T2。", updated_at: "2026-05-06T08:00:00Z", score: 0.8 },
        { id: "corrected", type: "episodic", content: "用户先选择虹桥机场T2，随后取消并改选浦东机场T1作为最终目的地。", updated_at: "2026-05-07T08:10:00Z", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const prompt = (await injector().execute(context(query)))[0]?.content ?? "";

    expect(prompt).toContain('cancelled_state="虹桥机场T2"');
    expect(prompt).toContain("cancelled_state_active=false");
    expect(prompt).toContain('final_state="浦东机场T1"');
    expect(prompt).toContain("STATE BOTH SIDES OF THE CORRECTION");
  });

  it("builds a fail-closed contract for a rescheduled appointment", async () => {
    const query = "轮胎检查改期以后，最终预约是什么？旧预约是否已经取消？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "old", type: "event", content: "轮胎检查先预约在8月26日上午9点。", updated_at: "2026-08-16T08:00:00Z", score: 0.8 },
        { id: "new", type: "event", content: "取消8月26日的检查，最终改到8月29日下午3点。", updated_at: "2026-08-17T08:00:00Z", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(query)))[0];
    expect(block?.metadata?.count).toBe(2);
    expect(block?.metadata?.evidenceStatusReasons).toEqual([]);
    expect(block?.metadata?.cockpitAnswerContract).toMatchObject({
      sufficient: true,
      requiredRelation: "cancelled",
      fallbackAnswer: expect.stringContaining("8月29日下午3点"),
    });
    expect(JSON.stringify(block?.metadata?.cockpitAnswerContract)).toContain("8月26日的检查");
  });

  it("routes current policies and entity comparisons without confusing entities for dates", () => {
    const currentCn = prepareCockpitQuery("如果现在电量只有10%，充电站选择规则是什么？顶棚和卫生间是否仍然优先？");
    const currentEn = prepareCockpitQuery("What is Maya's current restaurant profile after the medical update, including whether dairy is allowed?");
    const people = prepareCockpitQuery("比较驾驶员李和副驾阿岚的空调温度偏好，两个人分别是多少度？");
    const conditions = prepareCockpitQuery("Compare the audio policy when Noah drives alone with the policy when Emma is in the car.");

    expect(currentCn.reasons).toEqual(expect.arrayContaining(["profile-reference", "latest-final-update"]));
    expect(currentEn.reasons).toEqual(expect.arrayContaining(["profile-reference", "latest-final-update"]));
    expect(people.reasons).toContain("cross-session-synthesis");
    expect(people.reasons).not.toContain("multi-time-comparison");
    expect(conditions.reasons).toContain("cross-session-synthesis");
  });

  it("routes explicit named-subject field questions even without a history cue", () => {
    const english = prepareCockpitQuery("Which school does Emma attend?");
    const englishPreference = prepareCockpitQuery("What seat position does Maya prefer for long trips?");
    const chinese = prepareCockpitQuery("驾驶员使用哪一家充电网络的会员卡付款？");

    expect(english.reasons).toContain("subject-attribute-query");
    expect(englishPreference.reasons).toContain("subject-attribute-query");
    expect(chinese.reasons).toContain("subject-attribute-query");
  });

  it("treats a conditional priority exception as replacing the default priority", async () => {
    const query = "如果现在电量只有10%，充电站选择规则是什么？顶棚和卫生间是否仍然优先？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "default-priority", type: "persona", content: "默认选择有顶棚和卫生间的充电站。", score: 0.8 },
        { id: "priority", type: "persona", content: "通常优先有顶棚和卫生间的充电站；但电量低于15%时改为距离优先。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(query));

    expect(blocks[0]?.content).toContain("替换默认优先级");
    expect(blocks[0]?.content).toContain("默认优先项在该例外条件下不再是首要条件");
    expect(blocks[0]?.content).toContain("ANSWER THE PRIORITY YES/NO SUBQUESTION EXPLICITLY");
    expect(blocks[0]?.content).toContain('condition="电量低于15%时"');
    expect(blocks[0]?.content).toContain('condition_priority="距离"');
    expect(blocks[0]?.content).toContain('displaced_default="有顶棚和卫生间的充电站"');
    expect(blocks[0]?.content).toContain("displaced_default_still_primary=false");
  });

  it("grounds a low-energy priority rule from natural Chinese without exact question wording", async () => {
    const query = "续航告急，只剩8%。找补能位置时第一依据是什么，休息室和评分还排前面吗？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "priority", type: "persona", content: "用户于2026年10月11日说明：平时补能优先看休息室和评分；当续航低于15%时，最近距离优先，休息室和评分不再首要。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(query)))[0];
    expect(block?.metadata?.triggerReasons).toContain("latest-final-update");
    expect(block?.metadata?.cockpitAnswerContract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { value: "最近距离" },
        { value: "休息室和评分" },
      ]),
      fallbackAnswer: "最近距离优先；休息室和评分不再是首要条件。",
    });
  });

  it("grounds all spoken Chinese priority feature aliases against stable evidence slots", async () => {
    const query = "只有8%余电要进充电站，远近、休息室、口碑的优先级分别是什么？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "priority", type: "persona", content: "用户于2026年10月11日说明：平时补能优先看休息室和评分；当续航低于15%时，最近距离优先，休息室和评分不再首要。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    expect(extractCockpitNamedTargets(query)).toEqual(["距离", "休息室", "评分"]);
    const block = (await injector().execute(context(query)))[0];
    expect(block?.content).toContain('sufficient="true"');
    expect(block?.metadata?.cockpitAnswerContract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { value: "最近距离" },
        { value: "休息室和评分" },
      ]),
    });
  });

  it("assembles three Chinese occupant temperatures with stable ownership", async () => {
    const query = "车里有三个人：主驾顾原、副驾阿禾、后排孔老师各想要多少度？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "driver", type: "preference", content: "[顾原] 我是驾驶员时，座舱空调偏好21度。", score: 0.9 },
        { id: "front", type: "preference", content: "[阿禾] 我是副驾时，座舱空调偏好25度。", score: 0.9 },
        { id: "rear", type: "preference", content: "[孔老师] 我是后排乘客时，座舱空调偏好23度。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const contract = (await injector().execute(context(query)))[0]?.metadata?.cockpitAnswerContract;
    expect(contract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { label: "主驾顾原", value: "21度" },
        { label: "副驾阿禾", value: "25度" },
        { label: "后排孔老师", value: "23度" },
      ]),
    });
  });

  it("assembles compact and field-elliptical Chinese seat manifests without dropping an owner", async () => {
    const scenarios = [
      {
        query: "车内安排周遥主驾、小满前排右座、魏叔第二排，分别偏好几度？",
        names: ["周遥", "小满", "魏叔"],
      },
      {
        query: "主驾驶席是程野，副驾驶是小葵，后排是孔老师，三位设置各为何？",
        names: ["程野", "小葵", "孔老师"],
      },
      {
        query: "驾驶席里的周遥、邻座的小满、第二排里的魏叔请逐人查温度。",
        names: ["周遥", "小满", "魏叔"],
      },
      {
        query: "驾驶座上的苏木、右前方的宁宁、后舱里的魏叔逐一查询常用温度。",
        names: ["苏木", "宁宁", "魏叔"],
      },
      {
        query: "本趟乘员安排为林澈主驾驶位，豆豆副驾驶席，杜姐后排座，各人的温度设置是什么？",
        names: ["林澈", "豆豆", "杜姐"],
      },
      {
        query: "座次表是顾原主驾驶位、阿禾副驾驶位及孔老师后排座位，请逐人查询空调温度。",
        names: ["顾原", "阿禾", "孔老师"],
      },
      {
        query: "温言在驾驶座位、阿禾在前舱右座、杜姐在后舱座位，请核对每人的温度。",
        names: ["温言", "阿禾", "杜姐"],
      },
      {
        query: "本车座位绑定为许岚主驾位、小满前排右位、孟姨后排位，请把温度逐项配对。",
        names: ["许岚", "小满", "孟姨"],
      },
      {
        query: "座位表显示顾原主驾驶座、阿禾副驾驶座和孔老师后舱座，请逐个报空调温度。",
        names: ["顾原", "阿禾", "孔老师"],
      },
      {
        query: "驾驶座位是江屿，副驾位是豆豆，后舱座位是梁叔，请逐人给温度偏好。",
        names: ["江屿", "豆豆", "梁叔"],
      },
      {
        query: "座舱座次表列着程野驾驶席、小葵前排右位和孔老师第二排，请逐个回答温度。",
        names: ["程野", "小葵", "孔老师"],
      },
      {
        query: "三位乘员依次是苏木、宁宁、魏叔，座位对应主驾、副驾、后排，分别几度？",
        names: ["苏木", "宁宁", "魏叔"],
      },
      {
        query: "席位绑定显示许岚主驾位、小满前舱右座位、孟姨后舱座，请逐人回答温度偏好。",
        names: ["许岚", "小满", "孟姨"],
      },
    ];

    for (const scenario of scenarios) {
      vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
        code: 0,
        data: url.endsWith("/atomic/search") ? { items: [
          { id: "driver", type: "preference", content: `[${scenario.names[0]}] 我是驾驶员时，座舱空调偏好21度。`, score: 0.9 },
          { id: "front", type: "preference", content: `[${scenario.names[1]}] 我是副驾时，座舱空调偏好25度。`, score: 0.9 },
          { id: "rear", type: "preference", content: `[${scenario.names[2]}] 我是后排乘客时，座舱空调偏好23度。`, score: 0.9 },
        ] } : { messages: [] },
      }), { status: 200 })));

      const block = (await injector().execute(context(scenario.query)))[0];
      const contract = block?.metadata?.cockpitAnswerContract as CockpitAnswerContract | undefined;
      expect(block?.content, scenario.query).toContain('sufficient="true"');
      expect(contract?.fallbackAnswer, scenario.query).toBeTruthy();
      expect(contract?.requiredFacts, scenario.query).toEqual(expect.arrayContaining([
        expect.objectContaining({ value: "21度" }),
        expect.objectContaining({ value: "25度" }),
        expect.objectContaining({ value: "23度" }),
      ]));
    }
  });

  it("treats an ordered seat list as ownership context rather than a requested seat field", async () => {
    const query = "三位乘员为沈舟、宁宁、梁叔，座位依次驾驶席、前排右侧、后排，温度分别多少？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "driver", type: "preference", content: "[沈舟] 我是驾驶员时，座舱空调偏好22度。", score: 0.9 },
        { id: "front", type: "preference", content: "[宁宁] 我是副驾时，座舱空调偏好26度。", score: 0.9 },
        { id: "rear", type: "preference", content: "[梁叔] 我是后排乘客时，座舱空调偏好24度。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const contract = (await injector().execute(context(query)))[0]?.metadata?.cockpitAnswerContract;
    expect(contract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { label: "驾驶席沈舟", value: "22度" },
        { label: "前排右侧宁宁", value: "26度" },
        { label: "后排梁叔", value: "24度" },
      ]),
    });
  });

  it("assembles a bare three-name Chinese occupant list without inventing a fourth owner", async () => {
    const query = "综合历史偏好，这一车的林川、阿澄、方姨三个人分别调到几度？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "driver", type: "preference", content: "[林川] 我是驾驶员时，座舱空调偏好20度。", score: 0.9 },
        { id: "front", type: "preference", content: "[阿澄] 我是副驾时，座舱空调偏好24度。", score: 0.9 },
        { id: "rear", type: "preference", content: "[方姨] 我是后排乘客时，座舱空调偏好22度。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const contract = (await injector().execute(context(query)))[0]?.metadata?.cockpitAnswerContract;
    expect(extractChinesePersonTargets(query).map((item) => item.name)).toEqual(["林川", "阿澄", "方姨"]);
    expect(contract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { label: "林川", value: "20度" },
        { label: "阿澄", value: "24度" },
        { label: "方姨", value: "22度" },
      ]),
    });
  });

  it("contracts the complete Chinese media correction including the retained volume", async () => {
    const queries = [
      "按最后生效的夜间音频设置回答：听什么、最多几格、被替换项是否禁用？",
      "夜里听歌配置前后有变，请以最后版本回答当前类别、上限及旧类别去留。",
      "车机夜听现行配置要答齐，当前内容、音量限制及先前内容状态分别为何？",
    ];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "old", type: "preference", content: "用户于2026年10月14日设置夜间驾驶播放民谣，音量上限为11。", score: 0.8 },
        { id: "new", type: "preference", content: "用户于2026年10月18日更新夜间媒体：不再播放民谣，改为古典音乐；音量上限仍保持11。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    for (const query of queries) {
      const contract = (await injector().execute(context(query)))[0]?.metadata?.cockpitAnswerContract;
      expect(contract, query).toMatchObject({
        sufficient: true,
        requiredFacts: expect.arrayContaining([
          { label: "音乐类型", value: "古典音乐" },
          { label: "音量上限", value: "11" },
          { value: "民谣" },
        ]),
        fallbackAnswer: expect.stringContaining("旧音乐民谣已停用"),
      });
    }
  });

  it("fails closed when a Chinese media chain cannot project every requested correction fact", async () => {
    const query = "按最后生效的夜间音频设置回答：听什么、最多几格、被替换项是否禁用？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "old", type: "preference", content: "用户于2026年10月14日设置夜间驾驶播放民谣，音量上限为11。", score: 0.8 },
        { id: "new", type: "preference", content: "用户于2026年10月18日更新夜间音频为古典音乐，音量上限保持11。", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(query)))[0];
    const contract = block?.metadata?.cockpitAnswerContract as CockpitAnswerContract | undefined;
    expect(block?.metadata?.evidenceStatusReasons).toContain("answer_projection_media_transition_incomplete");
    expect(contract).toMatchObject({ sufficient: false, requiredFacts: [] });
    expect(contract?.fallbackAnswer).toBeUndefined();
    expect(block?.content).toContain('sufficient="false"');
    expect(block?.content).not.toContain("古典音乐");
  });

  it("contracts a terminal Chinese inspection cancellation and its prior final slot", async () => {
    const queries = [
      "综合全部年检变更，现在有效预约是什么；若为空，也要给出末次时段和替代状态。",
      "查清车检安排的最终落点：现有预约、撤销时段、替代预约分别是什么状态？",
      "年检更新链走完后是什么结果，末版档期仍有效吗，之后出现新安排没有？",
      "车辆检验排期最终版里保留、撤销、撤销后新建三类项目各怎样？",
    ];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "initial", type: "event", content: "用户于2026年10月6日将车辆年检预约在10月22日上午9点。", score: 0.7 },
        { id: "moved", type: "event", content: "用户于2026年10月10日取消10月22日的时段，改到10月25日下午3点。", score: 0.8 },
        { id: "cancelled", type: "event", content: "用户于2026年10月16日最终把10月25日的年检也取消，暂时不安排替代预约。", score: 0.9 },
        { id: "later-lumbar", type: "preference", content: "用户于2026年10月19日再次把长途腰托调到3档。", score: 0.95 },
      ] } : { messages: [] },
    }), { status: 200 })));

    for (const query of queries) {
      const contract = (await injector().execute(context(query)))[0]?.metadata?.cockpitAnswerContract as CockpitAnswerContract | undefined;
      expect(contract, query).toMatchObject({
        sufficient: true,
        requiredRelation: "cancelled",
        requiredFacts: expect.arrayContaining([
          { value: "10月25日下午3点" },
          expect.objectContaining({ value: "没有新预约" }),
        ]),
        fallbackAnswer: expect.stringContaining("年检已取消"),
      });
      expect(contract?.fallbackAnswer, query).toContain("当前没有有效年检预约");
    }
  });

  it("extracts an English 'becomes the priority' exception into the contract", async () => {
    const query = "The battery is at 9%. What is primary when choosing a charger, and are canopy and restroom still primary?";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "default", type: "persona", content: "Normally prioritize charging stations with a canopy and restroom.", updated_at: "2026-08-17T08:00:00Z", score: 0.8 },
        { id: "rule", type: "persona", content: "Normally prioritize charging stations with a canopy and restroom; when battery is below 12%, distance becomes the priority instead.", updated_at: "2026-08-18T08:00:00Z", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(query)))[0];
    expect(block?.metadata?.cockpitAnswerContract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([
        { value: "distance" },
        { value: "charging stations with a canopy and restroom" },
      ]),
    });
  });

  it("requires one explicit answer for every quoted alias target", async () => {
    const query = "目前‘老地方’和‘爸妈家’分别指哪里？请使用最新的别名定义。";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [] } : { messages: [
        { id: "alias-old", session_id: "alias-old", role: "user", content: "[source_time=2026-06-08T00:15:00Z] ‘老地方’以前指康平路88号。", timestamp: "2026-06-08T00:15:00Z", score: 0.03 },
        { id: "aliases", session_id: "alias-update", role: "user", content: "[source_time=2026-06-10T00:15:00Z] 更新：以后‘老地方’指公司地下车库；康平路88号只叫‘爸妈家’。", timestamp: "2026-06-10T00:15:00Z", score: 0.04 },
      ] },
    }), { status: 200 })));

    const prompt = (await injector().execute(context(query)))[0]?.content ?? "";

    expect(prompt).toContain('sufficient="true"');
    expect(prompt).toContain("OUTPUT ONE EXPLICIT VALUE FOR EVERY REQUESTED TARGET: 老地方, 爸妈家");
    expect(prompt).toContain('老地方="公司地下车库"');
    expect(prompt).toContain('爸妈家="康平路88号"');
    expect(prompt).toContain('required_output="老地方=公司地下车库；爸妈家=康平路88号"');
  });

  it("rejects named-person evidence that does not contain the requested attribute", async () => {
    const query = "Which school does Emma attend?";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "emma-audio", type: "persona", content: "When Emma is in the car, use children's audiobooks at volume 12.", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const prompt = (await injector().execute(context(query)))[0]?.content ?? "";

    expect(prompt).toContain('sufficient="false"');
    expect(prompt).toContain("no_matching_evidence");
    expect(prompt).not.toContain("children's audiobooks");
  });

  it("rejects Chinese same-person memories that do not contain the requested identifier field", async () => {
    const queries = [
      "记录里有没有江屿办公楼门卡编号？没有的话限定字段拒答。",
      "现有对话能提供程野办公门卡的准确编号吗？",
      "江屿办公楼门卡ID是多少？记录没覆盖就明确说不知道。",
      "请核对林澈公司门禁卡号，只有办公地点信息时应回答无法确定。",
    ];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "meeting", type: "persona", content: "[江屿] 平时会合点指清溪中心车库。", score: 0.9 },
        { id: "charging", type: "event", content: "[江屿] 到栖云能源点完成补能。", score: 0.8 },
        { id: "office", type: "persona", content: "[林澈] 办公地点在云港商务区。", score: 0.91 },
      ] } : { messages: [] },
    }), { status: 200 })));

    for (const query of queries) {
      const prompt = (await injector().execute(context(query)))[0]?.content ?? "";
      expect(prompt, query).toContain('sufficient="false"');
      expect(prompt, query).toContain("no_matching_evidence");
      expect(prompt, query).not.toContain("清溪中心车库");
      expect(prompt, query).not.toContain("栖云能源点");
      expect(prompt, query).not.toContain("云港商务区");
    }
  });

  it("rejects Chinese same-person media memories that do not contain the requested singer field", async () => {
    const queries = [
      "只凭现有对话可否确认陆青喜欢谁唱歌？音乐类别不能当作答案。",
      "以前是否记录了温言偏爱谁演唱？泛化的音乐风格不算人物证据。",
    ];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "music-type", type: "persona", content: "[陆青] 夜间偏好古典音乐。", score: 0.92 },
      ] } : { messages: [] },
    }), { status: 200 })));

    for (const query of queries) {
      const block = (await injector().execute(context(query)))[0];
      const contract = block?.metadata?.cockpitAnswerContract as CockpitAnswerContract | undefined;

      expect(block?.content, query).toContain('sufficient="false"');
      expect(block?.content, query).toContain("no_matching_evidence");
      expect(block?.content, query).not.toContain("古典音乐");
      expect(contract, query).toMatchObject({ sufficient: false, requiredFacts: [] });
      expect(contract?.fallbackAnswer, query).toBeUndefined();
    }
  });

  it("runs bounded L1+L0 expansion for all three old-failure query shapes", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({
      code: 0,
      data: {
        items: [{
          id: "m1",
          type: "event",
          content: "focused historical evidence: 充电事件; Maya tire inspection scheduled; work alias resolve",
          score: 0.9,
        }],
      },
    }), { status: 200, headers: { "content-type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    for (const query of OLD_FAILURE_QUERIES) {
      const blocks = await injector().execute(context(query));
      expect(blocks[0]?.content, query).toContain('sufficient="false"');
      expect(blocks[0]?.content, query).not.toContain("focused historical evidence");
      expect(blocks[0]?.content.length, query).toBeLessThan(3_000);
      expect(blocks[0]?.metadata?.triggerReasons, query).not.toEqual([]);
      expect(blocks[0]?.metadata?.searchedL0, query).toBe(true);
    }
    // Chinese aggregation=4 subqueries (original + two semantic canonical
    // forms + chain expansion), latest=2, two-date=4; every subquery searches
    // one L1 and one L0 endpoint, all in a single bounded parallel round.
    expect(fetchMock).toHaveBeenCalledTimes(20);
    expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/atomic/search"))).toHaveLength(10);
    expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/conversation/search"))).toHaveLength(10);
  });

  it("normalizes past local time and marks it for memory search", () => {
    const prepared = prepareCockpitQuery("昨天上午导航去了哪里？", {
      requestTime: REQUEST_TIME,
      timezone: "Asia/Shanghai",
    });

    expect(prepared.shouldSearchMemory).toBe(true);
    expect(prepared.requestLocalDate).toBe("2026-08-26");
    expect(prepared.retrievalText).toContain("2026-08-25 06:00–12:00");
    expect(prepared.timeEnvelope).toContain("不以记忆入库时间为基准");
    expect(prepareCockpitQuery("昨天去了哪里？", {
      requestTime: REQUEST_TIME,
      timezone: "Asia/Shanghai",
    }).resolutions[0]?.localRange).toBe("2026-08-25 00:00–24:00");
  });

  it("skips complete current commands with zero backend calls", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const blocks = await injector().execute(context("导航去上海虹桥站，避开高速。"));

    expect(blocks).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("injects future query time without an unnecessary history lookup", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const blocks = await injector().execute(context("明天上午九点提醒我充电。"));

    expect(fetchMock).not.toHaveBeenCalled();
    expect(blocks[0]?.content).toContain("2026-08-27 06:00–12:00");
    expect(blocks[0]?.content).not.toContain("tdai_recall_status");
  });

  it("recalls bounded Top-K evidence for an elliptical history request", async () => {
    const longContent = `上次选择深圳湾公园作为导航目的地，用户随后确认。${"证据".repeat(400)}`;
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toBe("http://memory-core:8420/v3/atomic/search");
      const body = JSON.parse(String(init.body)) as { query: string; limit: number };
      expect(body.query).toContain("上次那个");
      expect(body.limit).toBe(5);
      const headers = init.headers as Record<string, string>;
      expect(headers["x-tdai-service-id"]).toBe("vehicle-space-1");
      return new Response(JSON.stringify({
        code: 0,
        data: {
          items: [
            { id: "m1", type: "event", content: longContent, score: 0.92 },
            { id: "m2", type: "preference", content: "用户通常避开收费路段。", score: 0.74 },
          ],
        },
      }), { status: 200, headers: { "content-type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetchMock);

    const blocks = await injector().execute(context("还是导航去上次那个地方。"));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(blocks[0]?.content).toContain("Proxy 内部有界召回的座舱历史证据");
    expect(blocks[0]?.content).toContain("[source=l1 type=event score=0.920]");
    expect(blocks[0]?.content.length).toBeLessThan(2_500);
    expect(blocks[0]?.metadata?.mode).toBe("selective");
  });

  it("does not suggest a non-executable fallback when L1 misses", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      code: 0,
      data: { items: [] },
    }), { status: 200, headers: { "content-type": "application/json" } })));

    const blocks = await injector().execute(context("继续播放刚才那个。"));

    expect(blocks[0]?.content).toContain("没有召回到可用证据");
    expect(blocks[0]?.content).not.toContain("tdai_conversation_search");
    expect(blocks[0]?.content).not.toContain("curl");
    expect(blocks[0]?.content).toContain("不得猜测");
  });

  it("fails closed when every high-risk retrieval branch is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new TypeError("simulated memory backend outage");
    }));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[0]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain('sufficient="false"');
    expect(prompt).toContain("retrieval_incomplete_8_of_8");
    expect(prompt).toContain("必须明确回答无法从现有证据确定");
    expect(blocks[0]?.metadata?.retrievalAttempts).toBe(8);
    expect(blocks[0]?.metadata?.retrievalErrors).toBe(8);
  });

  it("fails closed on partial high-risk retrieval even when remaining evidence looks sufficient", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (url.endsWith("/conversation/search")) {
        throw new TypeError("simulated L0 branch outage");
      }
      return new Response(JSON.stringify({ code: 0, data: { items: [
        { id: "e1", type: "charging", content: "6月2日去了星港充电站。", score: 0.92 },
        { id: "e2", type: "charging", content: "6月4日去了星港充电站。", score: 0.89 },
        { id: "e3", type: "charging", content: "6月6日去了云桥充电站。", score: 0.86 },
      ] } }), { status: 200 });
    }));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[0]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain('timeline_points="3"');
    expect(prompt).toContain('sufficient="false"');
    expect(prompt).toContain("retrieval_incomplete_4_of_8");
    expect(blocks[0]?.metadata?.retrievalErrors).toBe(4);
  });

  it("turns a low-risk retrieval exception into an explicit no-evidence response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new TypeError("simulated memory backend outage");
    }));

    const blocks = await injector().execute(context("继续播放刚才那个。"));

    expect(blocks[0]?.content).toContain('matched="0"');
    expect(blocks[0]?.content).toContain("不得猜测");
    expect(blocks[0]?.metadata?.retrievalAttempts).toBe(1);
    expect(blocks[0]?.metadata?.retrievalErrors).toBe(1);
  });

  it("collects all explicitly counted events before allowing aggregation", async () => {
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body)) as { limit: number };
      expect(body.limit).toBe(13);
      if (url.endsWith("/atomic/search")) {
        return new Response(JSON.stringify({ code: 0, data: { items: [
          { id: "e1", type: "charging", content: "6月2日去了星港充电站。", score: 0.92 },
          { id: "e2", type: "charging", content: "6月4日去了星港充电站。", score: 0.89 },
          { id: "e3", type: "charging", content: "6月6日去了云桥充电站。", score: 0.86 },
        ] } }), { status: 200 });
      }
      return new Response(JSON.stringify({ code: 0, data: { messages: [] } }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[0]));

    expect(fetchMock).toHaveBeenCalledTimes(8);
    expect(blocks[0]?.content).toContain('sufficient="true"');
    expect(blocks[0]?.content).toContain('timeline_points="3"');
    expect(blocks[0]?.content).toContain("星港充电站: count=2");
    expect(blocks[0]?.content).toContain("云桥充电站: count=1");
    expect(blocks[0]?.content).toContain('highest_frequency="星港充电站"');
    expect(blocks[0]?.content).toContain("- count=2");
    expect(blocks[0]?.content).toContain('required_output="星港充电站: 2次"');
    expect(blocks[0]?.content).toContain("OUTPUT BOTH THE ENTITY AND COUNT: 星港充电站: 2 times");
    expect(blocks[0]?.metadata?.evidenceSufficient).toBe(true);
    expect(blocks[0]?.metadata?.cockpitAnswerContract).toMatchObject({
      version: 1,
      enforce: true,
      sufficient: true,
      requiredFacts: [{ value: "星港充电站" }, { value: "2" }],
      fallbackAnswer: "星港充电站，共2次。",
    });
  });

  it("honors Chinese event classifiers and projects common cockpit energy-site names", async () => {
    const query = "只统计这五条已完成的补能事件，出现次数最多的站点是哪一个？次数也要给出。";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "e1", type: "charging", content: "10月2日本次在松风补能港完成了充电。", updated_at: "2026-10-02T18:20:00+08:00", score: 0.92 },
        { id: "e2", type: "charging", content: "10月4日到松风补能港完成补能。", updated_at: "2026-10-04T18:20:00+08:00", score: 0.9 },
        { id: "e3", type: "charging", content: "10月6日在青岚补能中心完成了充电。", updated_at: "2026-10-06T18:20:00+08:00", score: 0.88 },
        { id: "e4", type: "charging", content: "10月8日到栖云能源点完成补能。", updated_at: "2026-10-08T18:20:00+08:00", score: 0.86 },
        { id: "e5", type: "charging", content: "10月10日本次在松风补能港完成了充电。", updated_at: "2026-10-10T18:20:00+08:00", score: 0.84 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(query)))[0];
    const prompt = block?.content ?? "";

    expect(prompt).toContain('timeline_points="5"');
    expect(prompt).toContain('sufficient="true"');
    expect(prompt).toContain("松风补能港: count=3");
    expect(prompt).toContain('highest_frequency="松风补能港"');
    expect(block?.metadata?.cockpitAnswerContract).toMatchObject({
      sufficient: true,
      requiredFacts: [{ value: "松风补能港" }, { value: "3" }],
      fallbackAnswer: "松风补能港，共3次。",
    });
  });

  it("keeps discourse-final aggregation in the frequency contract", async () => {
    const query = "盘点五次已完成充电，同一站合在一起，最终哪座场站占比最多、有几次？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "e1", type: "charging", content: "10月2日在松风补能港完成了充电。" },
        { id: "e2", type: "charging", content: "10月4日在松风补能港完成了充电。" },
        { id: "e3", type: "charging", content: "10月6日在青岚补能中心完成了充电。" },
        { id: "e4", type: "charging", content: "10月8日在栖云能源点完成了充电。" },
        { id: "e5", type: "charging", content: "10月10日在松风补能港完成了充电。" },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(query)))[0];

    expect(block?.metadata?.triggerReasons, query).toContain("aggregation-frequency");
    expect(block?.metadata?.triggerReasons, query).not.toContain("latest-final-update");
    expect(block?.content, query).toContain('sufficient="true"');
    expect(block?.content, query).toContain("松风补能港: count=3");
    expect(block?.content, query).not.toContain("final_update_relation_missing");
    expect(block?.metadata?.cockpitAnswerContract, query).toMatchObject({
      sufficient: true,
      fallbackAnswer: "松风补能港，共3次。",
    });
  });

  it("counts the common Chinese completion form '完成了车辆充电' as a grounded event", async () => {
    const query = "已完成的五笔补能事件里，最高频的站点是哪一个，共出现多少次？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "e1", type: "charging", content: "用户于2026年10月2日在青岚补能中心完成了车辆充电。" },
        { id: "e2", type: "charging", content: "用户于2026年10月4日在青岚补能中心完成了车辆充电。" },
        { id: "e3", type: "charging", content: "用户于2026年10月6日在松风补能港完成了车辆充电。" },
        { id: "e4", type: "charging", content: "用户于2026年10月8日在栖云能源点完成了车辆充电。" },
        { id: "e5", type: "charging", content: "用户于2026年10月10日在青岚补能中心完成了车辆充电。" },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(query)))[0];

    expect(block?.content).toContain('timeline_points="5"');
    expect(block?.content).toContain("青岚补能中心: count=3");
    expect(block?.metadata?.cockpitAnswerContract).toMatchObject({
      sufficient: true,
      fallbackAnswer: "青岚补能中心，共3次。",
    });
  });

  it("does not treat four records as complete when Chinese asks for five events", async () => {
    const query = "只统计这五条已完成的补能事件，出现次数最多的站点是哪一个？次数也要给出。";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "e1", type: "charging", content: "10月2日在松风补能港完成了充电。" },
        { id: "e2", type: "charging", content: "10月4日在松风补能港完成了充电。" },
        { id: "e3", type: "charging", content: "10月6日在青岚补能中心完成了充电。" },
        { id: "e4", type: "charging", content: "10月8日在栖云能源点完成了充电。" },
      ] } : { messages: [] },
    }), { status: 200 })));

    const prompt = (await injector().execute(context(query)))[0]?.content ?? "";

    expect(prompt).toContain('sufficient="false"');
    expect(prompt).toContain("aggregation_evidence_4_of_5");
  });

  it("orders a complete update timeline and exposes both initial and final states", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (url.endsWith("/atomic/search")) {
        return new Response(JSON.stringify({ code: 0, data: { items: [] } }), { status: 200 });
      }
      return new Response(JSON.stringify({ code: 0, data: { messages: [
        { id: "u2", role: "user", content: "On July 24 the inspection was finally moved to July 30.", timestamp: "2026-07-24T09:00:00Z", score: 0.02 },
        { id: "u1", role: "user", content: "On July 20 the tire inspection was scheduled for July 28.", timestamp: "2026-07-20T09:00:00Z", score: 0.03 },
      ] } }), { status: 200 });
    }));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[1]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt.indexOf("scheduled for July 28")).toBeLessThan(prompt.indexOf("finally moved to July 30"));
    expect(prompt).toContain('sufficient="true"');
    expect(prompt).toContain("不能只按相关度或入库时间猜最终状态");
  });

  it("projects a grounded cancellation as the final state and preserves the session tail", async () => {
    const query = "截至8月20日，Maya的保养预约最终还有效吗，还是已经取消？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (url.endsWith("/atomic/search")) {
        return new Response(JSON.stringify({ code: 0, data: { items: [] } }), { status: 200 });
      }
      if (url.endsWith("/conversation/query")) {
        return new Response(JSON.stringify({ code: 0, data: { messages: [
          { id: "initial", session_id: "maintenance", role: "user", content: "[maint-s01:001] [source_time=2026-08-10T08:00:00Z] Maya booked vehicle maintenance for August 22.", timestamp: "2026-08-10T08:00:00Z" },
          { id: "middle", session_id: "maintenance", role: "assistant", content: `Unrelated confirmation detail ${"x".repeat(900)}`, timestamp: "2026-08-11T08:00:00Z" },
          { id: "cancel", session_id: "maintenance", role: "user", content: "[maint-s01:003] [source_time=2026-08-18T08:00:00Z] Maya cancelled the maintenance appointment; it is no longer active.", timestamp: "2026-08-18T08:00:00Z" },
        ] } }), { status: 200 });
      }
      return new Response(JSON.stringify({ code: 0, data: { messages: [
        { id: "initial", session_id: "maintenance", role: "user", content: "[maint-s01:001] [source_time=2026-08-10T08:00:00Z] Maya booked vehicle maintenance for August 22.", timestamp: "2026-08-10T08:00:00Z", score: 0.03 },
        { id: "cancel", session_id: "maintenance", role: "user", content: "[maint-s01:003] [source_time=2026-08-18T08:00:00Z] Maya cancelled the maintenance appointment; it is no longer active.", timestamp: "2026-08-18T08:00:00Z", score: 0.02 },
      ] } }), { status: 200 });
    }));

    const blocks = await injector().execute(context(query));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain('sufficient="true"');
    expect(prompt).toContain("<tdai_grounded_final_state>");
    expect(prompt).toContain("relation=cancelled");
    expect(prompt).toContain("Maya cancelled the maintenance appointment");
    expect(prompt).toContain("must never be restated as active");
  });

  it("preserves the +08:00 calendar day and contracts Chinese date-point synonyms", async () => {
    const queries = [
      "会合地址有过临时有效期，请分别给出10月15日和10月19日当天对应的地点。",
      "同一个碰头口令在10月15日会导向哪里，在10月19日又会导向哪里？",
      "请按日期逐项列值：10月15日对应的碰头地；10月19日对应的碰头地。",
      "同一集合口令在10月15日与10月19日会给出哪两个目的地？",
      "同一个会合口令在10月15日和10月19日分别会导航到什么地方？",
      "10月15日约见应去何处，10月19日约见又应去何处？",
      "10月15日应去什么位置会面，到了10月19日应去哪里？",
      "同一个约见事项在10月15日、10月19日的有效位置分别是什么？",
      "请逐日返回会客落点，先答10月15日，再答10月19日。",
    ];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "base", type: "alias", content: "用户在2026年10月4日把‘会合点’设为青石研发院。" },
        { id: "temporary", type: "alias", content: "用户在2026年10月9日将‘会合点’临时变更为东岭智造园，有效期为10月12日至10月17日。" },
        { id: "restore", type: "alias", content: "用户于2026年10月18日早上把‘会合点’恢复到青石研发院。" },
        { id: "unrelated", type: "event", content: "用户于2026年10月19日仅更换仪表主题，并明确不调整导航。" },
      ] } : { messages: [] },
    }), { status: 200 })));

    for (const query of queries) {
      const block = (await injector().execute(context(query)))[0];
      const contract = block?.metadata?.cockpitAnswerContract;

      expect(block?.content, query).toContain('sufficient="true"');
      expect(block?.content, query).toContain('10月15日 -> "东岭智造园"');
      expect(block?.content, query).toContain('10月19日 -> "青石研发院"');
      expect(contract, query).toMatchObject({
        sufficient: true,
        requiredFacts: expect.arrayContaining([
          { label: "10月15日", value: "东岭智造园" },
          { label: "10月19日", value: "青石研发院" },
        ]),
        fallbackAnswer: "10月15日=东岭智造园；10月19日=青石研发院",
      });
    }
  });

  it("treats natural Chinese snapshot wording as a cutoff and excludes later state", async () => {
    const queries = [
      "站在10月16日这个时间点，按彼时记录，长途行驶腰托是多少档？",
      "记录截断在10月16日这一日时，座椅长途腰撑处于第几档？",
      "以不晚于10月16日为范围查询，跑长途时腰部支撑是多少档？",
      "按10月16日日终已有记录取值，之后更新不得影响长途支撑结果。",
      "本次查询纳入10月16日但屏蔽更晚事件，请给窗口末端的腰部支撑。",
      "记录时间线截至10月16日就停止，在该截点有效的远途支撑是多少？",
      "查询时间线只推进至10月16日，请回答截点上的远途支撑档数。",
      "把历史裁剪到10月16日当天结束，求裁剪后最后有效的腰托档数。",
      "只读取到10月16日当天结束的记录，长途腰托在这个右边界是几档？",
      "不采用晚于10月16日的记录，请查长途腰撑在当日终点的有效档数。",
    ];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "old", type: "seat", content: "用户于2026年10月8日把长途腰托设为1档。" },
        { id: "cutoff", type: "seat", content: "用户于2026年10月13日把长途腰托更新为5档。" },
        { id: "inspection-neighbor", type: "event", content: "用户于2026年10月16日最终把10月25日的年检取消，暂时不安排替代预约。" },
        { id: "future", type: "seat", content: "用户于2026年10月19日把长途腰托更新为3档。" },
      ] } : { messages: [] },
    }), { status: 200 })));

    for (const query of queries) {
      const block = (await injector().execute(context(query)))[0];

      expect(block?.content, query).toContain("更新为5档");
      expect(block?.content, query).not.toContain("年检取消");
      expect(block?.content, query).not.toContain("更新为3档");
      expect(block?.metadata?.cockpitAnswerContract, query).toMatchObject({
        sufficient: true,
        requiredFacts: expect.arrayContaining([{ value: "5档" }]),
        fallbackAnswer: "腰托档位=5档。",
      });
    }
  });

  it("refuses an implicit two-date request until both calendar points are named", async () => {
    const query = "查询双时点会客地点，两个日期的答案不能合并。";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "base", type: "alias", content: "用户在2026年10月4日把会合点设为青石研发院。" },
        { id: "temporary", type: "alias", content: "用户在2026年10月9日将会合点临时变更为东岭智造园，有效期为10月12日至10月17日。" },
        { id: "restore", type: "alias", content: "用户于2026年10月18日把会合点恢复到青石研发院。" },
      ] } : { messages: [] },
    }), { status: 200 })));

    const block = (await injector().execute(context(query)))[0];
    expect(block?.metadata?.evidenceSufficient).toBe(false);
    expect(block?.metadata?.evidenceStatusReasons).toContain("two_time_points_not_identified");
    const contract = block?.metadata?.cockpitAnswerContract as CockpitAnswerContract | undefined;
    expect(contract).toMatchObject({
      sufficient: false,
      requiredFacts: [],
    });
    expect(contract?.fallbackAnswer).toBeUndefined();
  });

  it("retrieves two date points separately and requires both to be covered", async () => {
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      if (url.endsWith("/atomic/search")) {
        return new Response(JSON.stringify({ code: 0, data: { items: [] } }), { status: 200 });
      }
      const body = JSON.parse(String(init.body)) as { query: string };
      const messages = body.query.includes("for september 10")
        ? [{ id: "d10", role: "user", content: "On September 10, work meant 100 Market Street.", timestamp: "2026-09-10T08:00:00Z", score: 0.02 }]
        : body.query.includes("for september 16")
          ? [{ id: "d16", role: "user", content: "On September 16, work meant 200 Harbor Road.", timestamp: "2026-09-16T08:00:00Z", score: 0.02 }]
          : [];
      return new Response(JSON.stringify({ code: 0, data: { messages } }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[2]));

    expect(fetchMock).toHaveBeenCalledTimes(8);
    expect(blocks[0]?.content).toContain("100 Market Street");
    expect(blocks[0]?.content).toContain("200 Harbor Road");
    expect(blocks[0]?.content).toContain("september 10 -> On September 10, work meant 100 Market Street");
    expect(blocks[0]?.content).toContain("september 16 -> On September 16, work meant 200 Harbor Road");
    expect(blocks[0]?.content).not.toContain("-> evidence #");
    expect(blocks[0]?.content).toContain("Never answer with 'evidence #N'");
    expect(blocks[0]?.content).toContain('sufficient="true"');
  });

  it("rejects a two-date answer when only the first date is present", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [] } : { messages: [
        { id: "d10", role: "user", content: "On September 10, work meant 100 Market Street.", timestamp: "2026-09-10T08:00:00Z", score: 0.02 },
      ] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[2]));

    expect(blocks[0]?.content).toContain('sufficient="false"');
    expect(blocks[0]?.content).toContain("missing_date:september 16");
  });

  it("forces abstention when an explicit three-event aggregation has only one event", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search")
        ? { items: [{ id: "only", type: "charging", content: "6月2日去了星港充电站。", score: 0.9 }] }
        : { messages: [] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[0]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain('sufficient="false"');
    expect(prompt).toContain("aggregation_evidence_1_of_3");
    expect(prompt).toContain("必须明确回答无法从现有证据确定");
    expect(prompt).not.toContain("tdai_conversation_search");
  });

  it("protects three aggregation event slots from L1 profile noise and duplicate confirmations", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "profile", type: "persona", content: "驾驶员通常偏好有顶棚的充电站。", score: 0.99 },
        { id: "alias", type: "episodic", content: "驾驶员更新了地点别名。", score: 0.98 },
        { id: "maintenance", type: "event", content: "6月7日车辆保养事件已记录。", score: 0.97 },
      ] } : { messages: [
        { id: "s04-user", role: "assistant", content: "[case-s04:001] [source_time=2026-06-05T13:00:00Z] 周五在虹桥枢纽充电站充了电。", timestamp: "2026-08-26T19:31:09Z", score: 0.032 },
        { id: "s04-confirm", role: "assistant", content: "[case-s04:002] [source_time=2026-06-05T13:00:00Z] 已记录虹桥枢纽充电站。", timestamp: "2026-08-26T19:31:10Z", score: 0.031 },
        { id: "s03-confirm", role: "assistant", content: "[case-s03:002] [source_time=2026-06-03T11:50:00Z] 已记录第二次在徐汇滨江超充站充电。", timestamp: "2026-08-26T19:31:11Z", score: 0.030 },
        { id: "s01-user", role: "assistant", content: "[case-s01:001] [source_time=2026-06-01T12:10:00Z] 周一去了徐汇滨江超充站充电。", timestamp: "2026-08-26T19:31:12Z", score: 0.029 },
      ] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[0]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain("case-s01:001");
    expect(prompt).toContain("case-s03:002");
    expect(prompt).toContain("case-s04:001");
    expect(prompt).not.toContain("通常偏好有顶棚");
    expect(prompt).not.toContain("车辆保养事件");
    expect(prompt).toContain("distinct_event=3");
    expect(prompt).toContain('timeline_points="3"');
    expect(prompt).toContain('sufficient="true"');
  });

  it("prioritizes occurrence sessions over higher-ranked generic confirmations", async () => {
    const query = "五次已完成的补能按地点汇总，众数站及次数是什么？";
    const queriedSessions: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body)) as { session_id?: string };
      if (url.endsWith("/atomic/search")) {
        return new Response(JSON.stringify({ code: 0, data: { items: [] } }), { status: 200 });
      }
      if (url.endsWith("/conversation/search")) {
        return new Response(JSON.stringify({ code: 0, data: { messages: [
          { id: "noise-1", session_id: "noise-settings", role: "assistant", content: "常态设置已记录。", score: 0.99 },
          { id: "noise-2", session_id: "noise-alias", role: "assistant", content: "地点别名已记录。", score: 0.98 },
          { id: "event-2", session_id: "event-2", role: "assistant", content: "已登记这次云杉能源点的补能。", score: 0.05 },
          { id: "event-3", session_id: "event-3", role: "assistant", content: "已登记这次云杉能源点的补能。", score: 0.04 },
          { id: "event-4", session_id: "event-4", role: "assistant", content: "已登记这次青岚补能中心的补能。", score: 0.03 },
          { id: "event-5", session_id: "event-5", role: "assistant", content: "已登记这次云杉能源点的补能。", score: 0.02 },
          { id: "event-6", session_id: "event-6", role: "assistant", content: "已登记这次松风补能港的补能。", score: 0.01 },
        ] } }), { status: 200 });
      }
      const session = body.session_id ?? "";
      queriedSessions.push(session);
      const station = session === "event-4"
        ? "青岚补能中心"
        : session === "event-6"
          ? "松风补能港"
          : "云杉能源点";
      const messages = session.startsWith("event-") ? [
        {
          id: `${session}-user`,
          session_id: session,
          role: "user",
          content: `[${session}:001] [source_time=2026-10-${session.slice(-1).padStart(2, "0")}T18:20:00+08:00] 本次在${station}完成了充电。`,
          timestamp: `2026-10-${session.slice(-1).padStart(2, "0")}T18:20:00+08:00`,
        },
      ] : [{ id: `${session}-noise`, session_id: session, role: "assistant", content: "无关设置。" }];
      return new Response(JSON.stringify({ code: 0, data: { messages } }), { status: 200 });
    }));

    const block = (await injector().execute(context(query)))[0];
    const prompt = block?.content ?? "";

    expect(queriedSessions).toEqual(expect.arrayContaining([
      "event-2", "event-3", "event-4", "event-5", "event-6",
    ]));
    expect(prompt).toContain('sufficient="true"');
    expect(prompt).toContain("云杉能源点: count=3");
    expect(block?.metadata?.cockpitAnswerContract).toMatchObject({
      sufficient: true,
      requiredFacts: expect.arrayContaining([{ value: "云杉能源点" }, { value: "3" }]),
    });
  });

  it("selects the entity-linked final timeline before truncation and excludes post-cutoff updates", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "initial-summary", type: "episodic", content: "Maya scheduled the tire inspection for July 28 at 9 a.m. on 21 July 2026.", updated_at: "2026-08-26T19:31:31Z", score: 0.033 },
        { id: "final-summary", type: "episodic", content: "Maya moved the tire inspection to July 30 at 2 p.m. and cancelled the old slot.", updated_at: "2026-08-26T19:31:32Z", score: 0.032 },
        { id: "noise", type: "episodic", content: "Maya checked tire pressure on July 5.", updated_at: "2026-08-26T19:31:30Z", score: 0.031 },
      ] } : { messages: [
        { id: "initial", role: "user", content: "[case-s11:001] [source_time=2026-07-21T02:00:00Z] Maya booked the tire inspection for July 28 at 9 a.m.", timestamp: "2026-08-26T19:31:40Z", score: 0.020 },
        { id: "final", role: "user", content: "[case-s12:001] [source_time=2026-07-24T06:00:00Z] Move the tire inspection to July 30 at 2 p.m.; cancel the old slot. [case-s12:002] The inspection is now July 30.", timestamp: "2026-08-26T19:31:41Z", score: 0.019 },
        { id: "future", role: "user", content: "[case-s13:001] [source_time=2026-07-26T06:00:00Z] Move the tire inspection again to August 2.", timestamp: "2026-08-26T19:31:42Z", score: 0.040 },
        { id: "food", role: "user", content: "[source_time=2026-07-08T04:00:00Z] Maya updated her food profile.", timestamp: "2026-08-26T19:31:20Z", score: 0.035 },
      ] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[1]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain("case-s11:001");
    expect(prompt).toContain("case-s12:001");
    expect(prompt).not.toContain("case-s13:001");
    expect(prompt).not.toContain("food profile");
    expect(prompt.indexOf("case-s11:001")).toBeLessThan(prompt.indexOf("case-s12:001"));
    expect(prompt).toContain('sufficient="true"');
  });

  it("keeps baseline, validity interval, and between-date restoration as one multi-time chain", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [] } : { messages: [
        { id: "base", role: "assistant", content: "[case-s03:001] [source_time=2026-08-03T00:00:00Z] Normal 'work' is Harbor Lab.", timestamp: "2026-08-26T19:31:13.505Z", score: 0.029 },
        { id: "temporary", role: "assistant", content: "[case-s05:001] [source_time=2026-08-31T08:00:00Z] From September 1 through September 14, 'work' resolves to Riverside Office.", timestamp: "2026-08-26T19:31:13.506Z", score: 0.033 },
        { id: "restore", role: "user", content: "[case-s06:001] [source_time=2026-09-15T09:00:00Z] The temporary project finished; change 'work' back to Harbor Lab.", timestamp: "2026-08-26T19:31:14.088Z", score: 0.031 },
        { id: "noise", role: "user", content: "[source_time=2026-10-01T03:00:00Z] October vegan food preference.", timestamp: "2026-08-26T19:31:14.090Z", score: 0.040 },
      ] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[2]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain("case-s03:001");
    expect(prompt).toContain("case-s05:001");
    expect(prompt).toContain("case-s06:001");
    expect(prompt).not.toContain("vegan food");
    expect(prompt).not.toContain("missing_date:");
    expect(prompt).not.toContain("between_dates_transition_missing");
    expect(prompt).toContain('sufficient="true"');
  });

  it("gates cross-session multi-person synthesis on at least two records", async () => {
    const query = "小雨在车上且要吃晚餐时，综合两个人在不同会话中的要求，应该选择哪家餐厅？";
    expect(prepareCockpitQuery(query).reasons).toContain("cross-session-synthesis");
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [] } : { messages: [
        { id: "s1", session_id: "trip-driver", role: "user", content: "驾驶员要求餐厅必须有停车位。", timestamp: "2026-08-20T09:00:00Z", score: 0.02 },
        { id: "s2", session_id: "trip-xiaoyu", role: "user", content: "小雨要求晚餐提供素食。", timestamp: "2026-08-22T09:00:00Z", score: 0.019 },
      ] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(query));

    expect(blocks[0]?.content).toContain('sufficient="true"');
    expect(blocks[0]?.content).toContain("停车位");
    expect(blocks[0]?.content).toContain("素食");
    expect(blocks[0]?.content).toContain('provenance_groups="2"');
    expect(blocks[0]?.metadata?.retrievalQueries).toBe(2);
  });

  it("rejects two-session synthesis when both sessions only cover one named person", async () => {
    const query = "比较驾驶员李和副驾阿岚的空调温度偏好，两个人分别是多少度？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [] } : url.endsWith("/conversation/query") ? { messages: [
        { id: "li-session", session_id: "trip-li-1", role: "user", content: "驾驶员李自己开车时空调偏好21度。", timestamp: "2026-08-20T09:00:00Z" },
      ] } : { messages: [
        { id: "li-1", session_id: "trip-li-1", role: "user", content: "驾驶员李自己开车时空调偏好21度。", timestamp: "2026-08-20T09:00:00Z", score: 0.03 },
        { id: "li-2", session_id: "trip-li-2", role: "user", content: "驾驶员李长途时仍然偏好21度。", timestamp: "2026-08-22T09:00:00Z", score: 0.02 },
      ] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(query));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain('sufficient="false"');
    expect(prompt).toContain("named_owner_coverage_1_of_2");
    expect(prompt).not.toContain('副驾阿岚的空调温度="21度"');
  });

  it("fails closed when source-session expansion is incomplete", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (url.endsWith("/conversation/query")) throw new TypeError("session expansion unavailable");
      return new Response(JSON.stringify({ code: 0, data: url.endsWith("/atomic/search") ? { items: [] } : { messages: [
        { id: "e1", session_id: "charge-session", role: "user", content: "6月2日去了星港充电站。", timestamp: "2026-06-02T08:00:00Z", score: 0.03 },
        { id: "e2", session_id: "charge-session", role: "user", content: "6月4日去了星港充电站。", timestamp: "2026-06-04T08:00:00Z", score: 0.02 },
        { id: "e3", session_id: "charge-session", role: "user", content: "6月6日去了云桥充电站。", timestamp: "2026-06-06T08:00:00Z", score: 0.01 },
      ] } }), { status: 200 });
    }));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[0]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain('sufficient="false"');
    expect(prompt).toContain("retrieval_incomplete_1_of_9");
    expect(prompt).not.toContain("6月2日去了星港充电站");
    expect(blocks[0]?.metadata?.supplementaryRetrievalErrors).toBe(1);
  });

  it("fails closed when an exact source-session event chain reaches the hard cap", async () => {
    const sessionMessages = Array.from({ length: 32 }, (_, index) => ({
      id: `session-row-${index}`,
      session_id: "long-charge-session",
      role: "user",
      content: `6月${index + 1}日记录了充电事件。`,
      timestamp: `2026-06-${String(index + 1).padStart(2, "0")}T08:00:00Z`,
    }));
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [] }
        : url.endsWith("/conversation/query") ? { messages: sessionMessages }
          : { messages: [
              { id: "e1", session_id: "long-charge-session", role: "user", content: "6月2日去了星港充电站。", timestamp: "2026-06-02T08:00:00Z", score: 0.03 },
              { id: "e2", session_id: "long-charge-session", role: "user", content: "6月4日去了星港充电站。", timestamp: "2026-06-04T08:00:00Z", score: 0.02 },
              { id: "e3", session_id: "long-charge-session", role: "user", content: "6月6日去了云桥充电站。", timestamp: "2026-06-06T08:00:00Z", score: 0.01 },
            ] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[0]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain('sufficient="false"');
    expect(prompt).toContain("retrieval_limit_saturated_1_of_9");
    expect(blocks[0]?.metadata?.saturatedSupplementaryBranches).toBe(1);
  });

  it("records semantic-search saturation without discarding complete explicit-count evidence", async () => {
    const thirteen = Array.from({ length: 13 }, (_, index) => ({
      id: `event-${index}`,
      type: "charging",
      content: `6月${index + 1}日去了星港充电站。`,
      score: 1 - index / 100,
    }));
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: thirteen } : { messages: [] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(OLD_FAILURE_QUERIES[0]));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain('sufficient="true"');
    expect(prompt).not.toContain("retrieval_limit_saturated_4_of_8");
    expect(blocks[0]?.metadata?.saturatedRetrievalBranches).toBe(4);
  });

  it("honors an explicit English event count before allowing aggregation", async () => {
    const query = "Across the three recorded charging visits, which station was used most often and how many times?";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "only", type: "charging", content: "On June 2 the driver charged at Star Harbor Charging Station.", score: 0.9 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(query));
    const prompt = blocks[0]?.content ?? "";

    expect(prompt).toContain('sufficient="false"');
    expect(prompt).toContain("aggregation_evidence_1_of_3");
  });

  it("expands recommendation questions with the selected option and explicit rationale", async () => {
    const query = "Which restaurant best matches Noah's current October food constraints, and why?";
    const searchQueries: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      const body = JSON.parse(String(init.body)) as { query?: string; session_id?: string };
      if (url.endsWith("/conversation/query")) {
        const messages = body.session_id === "profile"
          ? [{ id: "profile", session_id: "profile", role: "user", content: "For October Noah requires vegan Japanese food with no raw fish.", timestamp: "2026-10-01T03:00:00Z" }]
          : body.session_id === "dinner"
            ? [
                { id: "rationale", session_id: "dinner", role: "assistant", content: "[trip-s09:002] [source_time=2026-10-02T10:30:00Z] Miso Garden serves vegan Japanese ramen with no raw fish.", timestamp: "2026-10-02T10:30:00Z" },
                { id: "choice", session_id: "dinner", role: "user", content: "[trip-s09:003] [source_time=2026-10-02T10:30:00Z] Choose Miso Garden.", timestamp: "2026-10-02T10:30:00Z" },
              ]
            : [];
        return new Response(JSON.stringify({ code: 0, data: { messages } }), { status: 200 });
      }
      const queryText = body.query ?? "";
      searchQueries.push(queryText);
      const isExpansion = queryText.includes("selected restaurant or place") && url.endsWith("/conversation/search");
      const isFollowup = queryText.includes("[selected_option: Miso Garden]") && url.endsWith("/conversation/search");
      return new Response(JSON.stringify({ code: 0, data: url.endsWith("/atomic/search") ? { items: [] } : { messages: isFollowup ? [
        { id: "rationale", session_id: "dinner", role: "assistant", content: "[trip-s09:002] [source_time=2026-10-02T10:30:00Z] Miso Garden serves vegan Japanese ramen with no raw fish.", timestamp: "2026-10-02T10:30:00Z", score: 0.029 },
      ] : isExpansion ? [
        { id: "profile", session_id: "profile", role: "user", content: "For October Noah requires vegan Japanese food with no raw fish.", timestamp: "2026-10-01T03:00:00Z", score: 0.03 },
        { id: "choice", session_id: "dinner", role: "user", content: "[trip-s09:003] [source_time=2026-10-02T10:30:00Z] Choose Miso Garden.", timestamp: "2026-10-02T10:30:00Z", score: 0.028 },
      ] : [] } }), { status: 200 });
    }));

    const blocks = await injector().execute(context(query));
    const prompt = blocks[0]?.content ?? "";

    expect(searchQueries.some((value) => value.includes("explicit rationale"))).toBe(true);
    expect(prompt).toContain("Miso Garden serves vegan Japanese ramen with no raw fish");
    expect(prompt).toContain("Choose Miso Garden");
    expect(prompt).toContain('sufficient="true"');
    expect(blocks[0]?.metadata?.retrievalAttempts).toBeLessThanOrEqual(7);
    expect(blocks[0]?.metadata?.supplementaryRetrievalAttempts).toBe(2);
  });

  it("accepts grounded multi-person coverage when L1 provenance lacks session ids", async () => {
    const query = "比较驾驶员李和副驾阿岚的空调温度偏好，两个人分别是多少度？";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "li", type: "persona", content: "驾驶员李自己开车时空调偏好21度。", score: 0.9 },
        { id: "alan", type: "persona", content: "副驾阿岚坐车时空调偏好24度。", score: 0.89 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(query));

    expect(blocks[0]?.content).toContain("驾驶员李自己开车时空调偏好21度");
    expect(blocks[0]?.content).toContain("副驾阿岚坐车时空调偏好24度");
    expect(blocks[0]?.content).toContain('sufficient="true"');
    expect(blocks[0]?.content).toContain('驾驶员李的空调温度="21度"');
    expect(blocks[0]?.content).toContain('副驾阿岚的空调温度="24度"');
  });

  it("preserves condition-to-policy scope for passenger-dependent comparisons", async () => {
    const query = "Compare the audio policy when Noah drives alone with the policy when Emma is in the car.";
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "emma", type: "persona", content: "When Emma is in the car, use children's audiobooks and never set the volume above 12.", score: 0.9 },
        { id: "noah", type: "persona", content: "When Noah drives alone, use jazz at volume 20.", score: 0.89 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const blocks = await injector().execute(context(query));

    expect(blocks[0]?.content).toContain("条件 → 车辆应执行的策略");
    expect(blocks[0]?.content).toContain("条件中出现的人名不等于偏好主体");
    expect(blocks[0]?.content).toContain("When <condition>, use <vehicle policy>.");
    expect(blocks[0]?.content).toContain("Never use '<person> prefers ...'");
    expect(blocks[0]?.content).toContain('sufficient="true"');
  });

  it("injects generic user-facing, ownership, and field-scoped abstention rules", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      code: 0,
      data: { items: [] },
    }), { status: 200 })));

    const chinese = (await injector().execute(context("小雨在车上且要吃晚餐时，综合两个人在不同会话中的要求，应该选择哪家餐厅？为什么？")))[0]?.content ?? "";
    expect(chinese).toContain("禁止提及证据/编号/召回");
    expect(chinese).toContain("人物属性必须保持原归属");
    expect(chinese).toContain("仅缺所问字段时只说该字段无法确定");

    const english = (await injector().execute(context("Which school does Emma attend?")))[0]?.content ?? "";
    expect(english).toContain("never mention evidence numbers");
    expect(english).toContain("Keep each attribute with its owner");
    expect(english).toContain("only the requested field is missing");
  });

  it("emits cockpit curl guidance only with an executable shell loop", () => {
    const toolsInjector = new TdaiMemoryToolsInjector({
      proxyBaseUrl: "http://proxy:8096",
      domainProfile: "smart-cockpit",
    });
    expect(toolsInjector.cacheStrategy).toBe("none");
    expect(toolsInjector.execute(context("查一下历史"))).toEqual([]);

    const executable = context("查一下历史");
    executable.tools = [{ name: "Bash", description: "run shell", parameters: {} }];
    const blocks = toolsInjector.execute(executable);
    expect(blocks[0]?.content).toContain("<tdai_memory_tools>");
    expect(blocks[0]?.content).toContain("curl -sfk");
  });

  it("keeps prior tool-text cases free of command guidance with bounded growth", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
      code: 0,
      data: { items: [] },
    }), { status: 200, headers: { "content-type": "application/json" } })));
    const toolsInjector = new TdaiMemoryToolsInjector({
      proxyBaseUrl: "http://proxy:8096",
      domainProfile: "smart-cockpit",
    });

    for (const query of PRIOR_TOOL_TEXT_QUERIES) {
      const ctx = context(query);
      const blocks = [
        ...toolsInjector.execute(ctx),
        ...await injector().execute(ctx),
      ];
      const prompt = blocks.map((block) => block.content).join("\n");
      expect(prompt, query).not.toMatch(/curl|<tdai_memory_tools>|tdai_conversation_search/iu);
      expect(prompt.length, query).toBeLessThan(600);
    }
  });

  it("recognizes productive noun-first Chinese bounded-volume slots", () => {
    const cases: Array<[string, string[]]> = [
      ["夜听方案更正到末版后，现播音乐类型、音量最高值、旧内容效力各是什么？", ["音乐类型", "音量上限"]],
      ["夜间音频修订后，当前曲风、声量最大档位、原内容是否还有效？", ["曲风", "音量上限"]],
      ["晚间播放改版后，请答音乐类别、声音最高格数和老内容去留。", ["音乐类别", "音量上限"]],
      ["晚上听歌规则改完了，目前播放哪种音乐，音量最高多少，旧内容是否停播？", ["音乐类型", "音量上限"]],
      ["音响设置更新后，现在听哪类，声量最大几档，原来的还播吗？", ["音乐类型", "音量上限"]],
      ["夜间播放方案末版怎样，现用内容类型、声音最高值、原内容去留分别为何？", ["音乐类型", "音量上限"]],
      ["夜间播放定版后，新音乐类型、声音最大值、被替换内容状态分别是什么？", ["音乐类型", "音量上限"]],
      ["夜驾媒体收口后，当前音乐类别、声量的最高格位、旧门类效力各为何？", ["音乐类别", "音量上限"]],
    ];

    for (const [query, namedTargets] of cases) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.domain, query).toBe("media");
      expect(semantics.targets.map((target) => target.key), query).toEqual(expect.arrayContaining([
        "music-type", "volume-limit",
      ]));
      expect(semantics.intents, query).toEqual(expect.arrayContaining([
        "latest-state", "correction-state", "multi-target-state",
      ]));
      expect(extractCockpitNamedTargets(query), query).toEqual(namedTargets);
    }
  });

  it("recognizes date-first Chinese permitted-visibility boundaries", () => {
    const queries = [
      "将10月16日设为允许读取的最晚日期，长途腰部支撑那时怎样设置？",
      "把10月16日定为本次许可查看的最后一天，腰托当时是几档？",
      "10月16日设成可采用的最后时点，远途座椅支撑值是多少？",
      "记忆最多开放至10月16日日末，之后内容忽略，腰部支撑当时是多少？",
    ];

    for (const query of queries) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.domain, query).toBe("lumbar");
      expect(semantics.dateMentions, query).toHaveLength(1);
      expect(semantics.intents, query).toEqual(expect.arrayContaining([
        "latest-state", "cutoff-state",
      ]));
      expect(semantics.targets.map((target) => target.key), query).not.toContain("music-type");
    }
  });

  it("compiles productive meeting interrogatives, explicit endpoints, and ordered seat lists", () => {
    for (const query of [
      "10月15日当天碰头去什么地址，10月19日当天碰头又去什么地址？",
      "同一碰头计划跨两个日期，10月15日应去哪里，10月19日应去哪里？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.domain, query).toBe("meeting-point");
      expect(semantics.targets.map((target) => target.key), query).toContain("meeting-point");
      expect(semantics.intents, query).toContain("two-date-state");
    }

    for (const query of [
      "规定10月16日为历史终点，更晚内容不要用，长途座椅支撑是多少档？",
      "以10月16日日末为历史终点，屏蔽之后修改，远途腰撑落在哪一档？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.domain, query).toBe("lumbar");
      expect(semantics.intents, query).toEqual(expect.arrayContaining(["latest-state", "cutoff-state"]));
    }

    expect(extractChinesePersonTargets(
      "三位乘员分别为苏木、宁宁、魏叔，席位依次主驾驶位、副驾驶席、后排座，逐个回答几度。",
    )).toEqual([
      { name: "苏木", role: "driver", roleLabel: "主驾驶位" },
      { name: "宁宁", role: "front-passenger", roleLabel: "副驾驶席" },
      { name: "魏叔", role: "rear-passenger", roleLabel: "后排座" },
    ]);
    expect(extractChinesePersonTargets(
      "乘员名单为许岚、小满、孟姨，座次对应驾驶席、副驾位、后舱座，分别查温度。",
    )).toEqual([
      { name: "许岚", role: "driver", roleLabel: "驾驶席" },
      { name: "小满", role: "front-passenger", roleLabel: "副驾位" },
      { name: "孟姨", role: "rear-passenger", roleLabel: "后舱座" },
    ]);
  });

  it("compiles productive Chinese seat manifests, modifier-first aliases, and dashboard reports", () => {
    const seatCases: Array<[string, Array<[string, string]>]> = [
      [
        "席位登记写着林澈驾驶席、豆豆右前座、杜姐第二排座，三人各常设几度？",
        [["林澈", "driver"], ["豆豆", "front-passenger"], ["杜姐", "rear-passenger"]],
      ],
      [
        "座舱绑定为周遥驾驶座位、小满前排右席、魏叔后舱座，请分别给空调设置。",
        [["周遥", "driver"], ["小满", "front-passenger"], ["魏叔", "rear-passenger"]],
      ],
      [
        "座次单列出陆青主驾席位、可可副驾座位、孟姨后排乘员，各自习惯几度？",
        [["陆青", "driver"], ["可可", "front-passenger"], ["孟姨", "rear-passenger"]],
      ],
      [
        "车上安排江屿坐驾驶位、豆豆坐前舱右座、梁叔坐后排座，逐一回答温度。",
        [["江屿", "driver"], ["豆豆", "front-passenger"], ["梁叔", "rear-passenger"]],
      ],
      [
        "座位表把许岚放在主驾驶座、小满放在副驾驶位、孟姨放在第二排，请按人物查几度。",
        [["许岚", "driver"], ["小满", "front-passenger"], ["孟姨", "rear-passenger"]],
      ],
      [
        "座次绑定显示顾原主驾驶座位、阿禾副驾驶席位、孔老师后排座位，请逐人给温度。",
        [["顾原", "driver"], ["阿禾", "front-passenger"], ["孔老师", "rear-passenger"]],
      ],
    ];

    for (const [query, expected] of seatCases) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.domain, query).toBe("occupant-temperature");
      expect(semantics.people.map((person) => [person.name, person.role]), query).toEqual(expected);
    }

    for (const query of [
      "导航口令最后绑定为何，惯用停车点、亲属住所、看诊位置逐项还原。",
      "把最终地点配对答全：常用泊车位置、家属住所、门诊处分别是什么地址？",
    ]) {
      const semantics = compileChineseCockpitSemantics(query);
      expect(semantics.domain, query).toBe("navigation-alias");
      expect(semantics.targets.map((target) => target.key), query).toEqual([
        "parking-alias", "relative-home-alias", "clinic-alias",
      ]);
      expect(extractCockpitNamedTargets(query), query).toEqual([
        "固定车位", "亲友住处", "诊所",
      ]);
    }

    const priorityQuery = "仪表报10%要紧急补电，距离是否置首，休息室和高分因素怎样安排？";
    const priority = compileChineseCockpitSemantics(priorityQuery);
    expect(priority, priorityQuery).toMatchObject({ domain: "charging-priority" });
    expect(priority.intents, priorityQuery).toEqual(expect.arrayContaining([
      "conditional-priority", "latest-state",
    ]));
  });

  it("registers selective L1 recall in the live injection pipeline", async () => {
    const config = structuredClone(DEFAULT_CONFIG);
    config.injection.enabled = true;
    config.injection.injectors = ["tdai-memory"];
    config.injection.externalGatewayUrl = "http://proxy:8096";
    config.storage.enabled = true;
    config.storage.backend = "memory";
    config.tdai.enabled = true;
    config.tdai.endpoint = memoryConfig.endpoint;
    config.tdai.apiKey = memoryConfig.apiKey;
    config.tdai.memory = {
      enabled: true,
      inject: true,
      domainProfile: "smart-cockpit",
      timezone: "Asia/Shanghai",
      writeL0: true,
      recallL1: true,
      injectL2L3: false,
      l1Limit: 5,
      l2Limit: 3,
      timeoutMs: 1_000,
    };
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({
      code: 0,
      data: { items: [{ id: "m1", type: "event", content: "上次导航到了深圳湾公园。", score: 0.9 }] },
    }), { status: 200, headers: { "content-type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await getInjectionPipeline(config).process({
      model: "small-model",
      messages: [{ role: "user", content: "还是去上次那个地方。" }],
    }, context("unused").metadata);

    expect(JSON.stringify(result)).toContain("上次导航到了深圳湾公园");
    expect(JSON.stringify(result)).toContain("tdai_recalled_l1_memories");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://memory-core:8420/v3/atomic/search");
  });

  it("promotes the high-risk answer contract through the live pipeline metadata", async () => {
    const config = structuredClone(DEFAULT_CONFIG);
    config.injection.enabled = true;
    config.injection.injectors = ["tdai-memory"];
    config.injection.externalGatewayUrl = "http://proxy:8096";
    config.storage.enabled = true;
    config.storage.backend = "memory";
    config.tdai.enabled = true;
    config.tdai.endpoint = memoryConfig.endpoint;
    config.tdai.apiKey = memoryConfig.apiKey;
    config.tdai.memory = {
      enabled: true,
      inject: true,
      domainProfile: "smart-cockpit",
      timezone: "Asia/Shanghai",
      writeL0: true,
      recallL1: true,
      injectL2L3: false,
      l1Limit: 5,
      l2Limit: 3,
      timeoutMs: 1_000,
    };
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify({
      code: 0,
      data: url.endsWith("/atomic/search") ? { items: [
        { id: "e1", type: "charging", content: "6月2日去了星港充电站。", score: 0.92 },
        { id: "e2", type: "charging", content: "6月4日去了星港充电站。", score: 0.89 },
        { id: "e3", type: "charging", content: "6月6日去了云桥充电站。", score: 0.86 },
      ] } : { messages: [] },
    }), { status: 200 })));

    const metadata = context("unused").metadata;
    await getInjectionPipeline(config).process({
      model: "small-model",
      messages: [{ role: "user", content: OLD_FAILURE_QUERIES[0] }],
    }, metadata);

    expect(metadata.custom?.cockpitAnswerContract).toMatchObject({
      version: 1,
      enforce: true,
      sufficient: true,
      fallbackAnswer: "星港充电站，共2次。",
    });
  });
});
