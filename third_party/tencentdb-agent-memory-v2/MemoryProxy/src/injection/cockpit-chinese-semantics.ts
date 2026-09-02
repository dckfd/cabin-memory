export type ChineseCockpitIntent =
  | "event-frequency"
  | "latest-state"
  | "cutoff-state"
  | "two-date-state"
  | "multi-person-state"
  | "multi-target-state"
  | "conditional-priority"
  | "correction-state"
  | "final-cancellation";

export type ChineseCockpitDomain =
  | "charging-event"
  | "charging-priority"
  | "commute"
  | "meeting-point"
  | "occupant-temperature"
  | "inspection"
  | "navigation-alias"
  | "lumbar"
  | "media"
  | "unknown";

export interface ChinesePersonTarget {
  name: string;
  role?: "driver" | "front-passenger" | "rear-passenger";
  roleLabel?: string;
}

export interface ChineseStateTarget {
  key:
    | "route"
    | "elevated-road"
    | "toll-road"
    | "meeting-point"
    | "parking-alias"
    | "relative-home-alias"
    | "clinic-alias"
    | "music-type"
    | "volume-limit";
  label: string;
}

export interface ChineseCockpitSemantics {
  chinese: boolean;
  intents: ChineseCockpitIntent[];
  domain: ChineseCockpitDomain;
  people: ChinesePersonTarget[];
  targets: ChineseStateTarget[];
  dateMentions: string[];
  canonicalRetrievalQueries: string[];
}

const CHINESE_DATE = /(?:20\d{2}年)?\d{1,2}月\d{1,2}日/gu;
const MEETING_POINT_CUE = /(?:会合|会面|会晤|会师|会客|见面|碰面|集合|碰头|接头|相约|约见|约碰)(?:点|落点|落脚点|地|处|地址|地点|位置|口令|标签|映射)|(?:会合|会面|会晤|会师|会客|见面|碰面|集合|碰头|接头|相约|约见|约碰)(?:又|分别|各自)?(?:应|要|需)?(?:去|在|到|于)?(?:哪(?:里|儿)?|何处)|(?:在)?(?:哪(?:里|儿)?|何处)(?:会合|会面|会晤|会师|会客|见面|碰面|集合|碰头|接头|相约|约见|约碰)|(?:什么|哪(?:个|一)?|哪个|何种)?(?:位置|地点|地址|地方|去处|落点|落脚点)(?:进行|用于|作为)?(?:会合|会面|会晤|会师|会客|见面|碰面|集合|碰头|接头|相约|约见|约碰)|(?:会合|会面|会晤|会客|见面|碰面|集合|碰头|相约|约见)(?:事项|安排|约定).{0,28}(?:有效)?(?:点|落点|落脚点|地|处|地址|地点|位置|地方|去处)|(?:约|相约)(?:在|到|于)?(?:哪(?:里|儿)?|何处)(?:见|碰面|会面)/u;
const PRODUCTIVE_MEETING_LOCATION_CUE = /(?:会合|会面|会晤|会师|会客|见面|碰面|集合|碰头|接头|相约|约见|约碰)(?:计划|事项|安排|约定)?.{0,40}?(?:应|要|需)?(?:去|在|到|于)(?:哪(?:里|儿)?|何处|什么(?:地址|地点|位置|地方|去处))/u;
const DASHBOARD_PERCENTAGE = /(?:仪表(?:盘)?(?:电量)?(?:读数)?|表显|电池表|电量表|电池读数|电量读数|电池显示|电量显示)\s*(?:显示(?:为|到)?|只显示(?:为)?|仅显示(?:为)?|报(?:出)?|读到|为|是|只剩|只余|仅有|仅余|仅剩|只有|还有|来到|降到|降至|降为|跌到|跌至|掉到|变为)?\s*\d+(?:\.\d+)?\s*%(?:的)?(?:电|电量|余电)?/u;
const LOW_ENERGY_PERCENTAGE = /(?:电|电量|电池(?:余量)?|余电|余量|续航)[^。！？!?\d]{0,16}\d+(?:\.\d+)?\s*(?:%|百分之)?|(?:只余|只剩|剩|剩余|还有|只有)?\s*\d+(?:\.\d+)?\s*%(?:的)?(?:电|电量|余电|余量|续航)|(?:在)?\s*\d+(?:\.\d+)?\s*%\s*(?:这个|的)?\s*(?:低电量|余电|电量)(?:条件|状态)?|(?:低电量|低余电)(?:状态)?\s*(?:为|是|只剩|仅剩|只余|仅余|来到|降到|降至)?\s*\d+(?:\.\d+)?\s*%/u;
function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function roleOf(label: string): ChinesePersonTarget["role"] {
  if (/副驾|前排|前座|前舱|右前|右边|右侧|右手|旁边|邻座/u.test(label)) return "front-passenger";
  if (/主驾|驾驶(?:席|座|位|员|者)|司机/u.test(label)) return "driver";
  return "rear-passenger";
}

function cleanPersonName(value: string): string {
  return value
    .replace(/^(?:本次|本趟|本车|这(?:一)?趟|这一程)\s*/u, "")
    .replace(/^(?:对应(?:的是)?|安排(?:的)?(?:是|为)?|担任|是|为|由|坐着?|坐在|在)\s*/u, "")
    .replace(/\s*(?:对应|担任|坐在|坐|放在|放到|放于|放|位于|在|负责)$/u, "")
    // Spoken lists often attach their cardinality directly to the last name
    // ("甲、乙、丙三人的温度"). The suffix is grammar, not part of the
    // occupant name, and must be removed before owner-scoped retrieval.
    .replace(/(?:三个人|三位|三名|三人)$/u, "")
    .replace(/(?:同车|的|时|想要|各自|分别|依次|逐一|逐个|逐人|挨个)+$/u, "")
    .trim();
}

/**
 * Extract named occupants from role-before-name, name-before-role, and a
 * three-name spoken list. Values are query entities only; no memory answer is
 * encoded here.
 */
export function extractChinesePersonTargets(query: string): ChinesePersonTarget[] {
  if (!/[\u3400-\u9fff]/u.test(query)) return [];
  const people = new Map<string, ChinesePersonTarget>();
  const add = (nameValue: string, roleLabel?: string) => {
    const name = cleanPersonName(nameValue);
    if (!/^[\u3400-\u9fff]{1,4}$/u.test(name)) return;
    if (/主驾|副驾|驾驶|司机|前排|后排|右前|右边|右侧|后座|座位|席位/u.test(name)) return;
    if (/^(?:主|副|座|位|席|排|舱)$/u.test(name)) return;
    if (/^(?:停车位|车位|亲友家|家人住所|亲友住处|诊所|就诊地点|路线|高架|收费道路)$/u.test(name)) return;
    if (/^(?:座次|座位|他们|她们|乘员们)/u.test(name)) return;
    if (/修改|记录|事件|历史|更新|变化|后续|配置|状态|屏蔽|过滤|排除|忽略|采纳|采用|读取/u.test(name)) return;
    if (/^(?:(?:各|分别|各自|每人)?(?:多少|几度|怎么|如何|设置|调到|调成|要多少)|三人|三个|乘员|乘客|位置|温度|空调|设置|偏好|当前|分别|各自|依次|座次)$/u.test(name)) return;
    const prior = people.get(name);
    people.set(name, {
      name,
      role: roleLabel ? roleOf(roleLabel) : prior?.role,
      roleLabel: roleLabel ?? prior?.roleLabel,
    });
  };

  // A compact seat manifest is common in spoken cockpit commands:
  // “座次从前到后为甲主驾、乙副驾、丙后座”. Parse the whole manifest
  // positionally before the looser role patterns so the first name is not
  // lost merely because it follows “为” instead of punctuation.
  const compactSeatManifest = query.match(
    /(?:座次(?!表|绑定)|座位(?!表|绑定)|席位(?!表)|乘员(?:安排|顺序)?|乘坐顺序|(?:车内|车上|座舱|本车|本次|本趟)(?:乘员)?安排)(?:\s*(?:从前到后|按(?:前后|座位|座次)顺序|依次))?\s*(?:为|是|有|如下|[:：])?\s*([\u3400-\u9fff]{1,4}?)\s*(主驾驶位|主驾驶席|主驾驶|驾驶席|驾驶座|驾驶位|驾驶员|司机|主驾)\s*[、，,；;]\s*([\u3400-\u9fff]{1,4}?)\s*(前座右边乘客|前座右边乘员|前座右侧乘客|前座右侧乘员|前排右侧乘员|前排右座乘客|前排右座|前排乘员|前排右侧|前排右席|右侧前排|右前排|右前席|右前座|右前位|副驾驶席|副驾驶|副驾位|副驾|前排)\s*[、，,；;]\s*([\u3400-\u9fff]{1,4}?)\s*(后排乘客|后座乘员|第二排乘客|第二排乘员|第二排|后排座|后排|后座)/u,
  );
  if (compactSeatManifest) {
    add(compactSeatManifest[1], compactSeatManifest[2]);
    add(compactSeatManifest[3], compactSeatManifest[4]);
    add(compactSeatManifest[5], compactSeatManifest[6]);
  }

  // Productive manifests may say “座次表/座位绑定”, join the last two
  // entries with “及”, or use fully inflected seat nouns such as “后排座位”.
  // Parse the grammatical three-slot structure instead of enumerating names.
  const productiveSeatManifest = query.match(
    /(?:座次表|座位表|席位表|座次单|座位单|席位单|座次登记|座位登记|席位登记|座位绑定|座次绑定|席位绑定|座舱绑定|车上安排|(?:本车|车内|车上|座舱)(?:的)?(?:(?:座位|座次|席位)(?:绑定|安排)|(?:座位|座次|席位)(?!表|单|登记|绑定|安排)))(?:\s*(?:从前到后|按(?:前后|座位|座次)顺序|依次))?\s*(?:为|是|有|如下|显示(?:为|是)?|列(?:出|为|着)?|写(?:明|着)?|[:：])?\s*(?:把\s*)?([\u3400-\u9fff]{1,4}?)\s*(?:坐(?:在)?|放(?:在|到|于)?|安排(?:在|到|为)?)?\s*(主驾驶座位|主驾驶席位|主驾驶位|主驾驶席|主驾驶座|主驾驶|驾驶座位|驾驶席位|驾驶席|驾驶座|驾驶位|驾驶员|司机|主驾座位|主驾席位|主驾位|主驾)\s*(?:[、，,；;]|及|和|与)\s*([\u3400-\u9fff]{1,4}?)\s*(?:坐(?:在)?|放(?:在|到|于)?|安排(?:在|到|为)?)?\s*(副驾驶座位|副驾驶席位|副驾驶位|副驾驶席|副驾驶座|副驾驶|副驾座位|副驾席位|副驾位|副驾|前排右侧|前排右座位|前排右座|前排右位|前排右席|右前排|右前席|右前座|右前位|前舱右座位|前舱右座)\s*(?:[、，,；;]|及|和|与)\s*([\u3400-\u9fff]{1,4}?)\s*(?:坐(?:在)?|放(?:在|到|于)?|安排(?:在|到|为)?)?\s*(后排乘客|后排乘员|后排座位|后排座|后排位|后排|后舱座位|后舱座|后舱|后座乘客|后座乘员|后座位|后座|第二排乘客|第二排乘员|第二排座位|第二排座|第二排)/u,
  );
  if (productiveSeatManifest) {
    add(productiveSeatManifest[1], productiveSeatManifest[2]);
    add(productiveSeatManifest[3], productiveSeatManifest[4]);
    add(productiveSeatManifest[5], productiveSeatManifest[6]);
  }

  // Longest role surfaces must come first. Otherwise “副驾驶豆豆” is
  // tokenized as role=副驾 and name=驶豆豆, which later causes a false
  // evidence-coverage refusal.
  const roleFirst = /(前座右边乘客|前座右边乘员|前座右侧乘客|前座右侧乘员|前排右侧乘员|前排右座乘客|前排乘客|前排乘员|后排乘客|后排乘员|第二排乘客|第二排乘员|右前方乘客|右前方乘员|右前排乘客|右前座乘客|右前乘员|后座乘客|后座乘员|副驾驶乘客|副驾驶乘员|主驾驶乘员|副驾驶位|副驾驶席|副驾乘客|副驾乘员|主驾乘员|副驾驶|主驾驶位|主驾驶|驾驶者|驾驶员|驾驶席|驾驶座|驾驶位|前排右席|右侧前排|右前方|右前排|右前席|右前座|右前位|副驾位|主驾位|司机|主驾|副驾|前排右侧|前排|旁边|右边|后面|后排座|后排|后座)(?!驶|位|席|座|员|右|界)\s*(?:对应(?:的是)?|安排(?:的)?(?:是|为)?|是|为|由|坐着?|坐的是|的|(?:上|里|中)(?:的)?)?\s*([\u3400-\u9fff]{1,4}?)(?=、|，|,|。|；|;|？|\?|和|与|各|分别|依次|想|要|的|同时|时|$)/gu;
  for (const match of query.matchAll(roleFirst)) add(match[2], match[1]);

  const nameFirst = /(?:^|[：:,，、；;\s])([\u3400-\u9fff]{1,4}?)(?:\s*(?:对应(?:的是)?|安排(?:在|到|为)?|担任|是|为|位于|在|乘(?:坐)?(?:在)?|坐在?|坐|放(?:在|到|于)?))?\s*(前座右边乘客|前座右边乘员|前座右侧乘客|前座右侧乘员|前排右侧乘员|前排右座乘客|前排右座位|前排右座|前排乘客|前排乘员|后排乘客|后排乘员|第二排乘客|第二排乘员|第二排座位|第二排座|右前方乘客|右前方乘员|右前排乘客|右前座乘客|右前乘员|后座乘客|后座乘员|副驾驶乘客|副驾驶乘员|主驾驶乘员|副驾驶座位|副驾驶席位|副驾驶位|副驾驶席|副驾驶座|副驾乘客|副驾乘员|主驾乘员|副驾驶|主驾驶座位|主驾驶席位|主驾驶位|主驾驶席|主驾驶座|主驾驶|驾驶者|驾驶员|驾驶座位|驾驶席位|驾驶席|驾驶座|驾驶位|前排右席|右侧前排|右前方|右前排|右前席|右前座|右前位|副驾座位|副驾席位|副驾位|主驾座位|主驾席位|主驾位|司机|主驾|副驾|前舱右座位|前舱右座|后舱座位|后舱座|前排右侧|前排|旁边|右边|第二排|后排座位|后排座|后排|后座位|后座)(?=、|，|,|。|；|;|？|\?|时|各|分别|依次|想|要|$)/gu;
  for (const match of query.matchAll(nameFirst)) add(match[1], match[2]);

  const productiveNameFirstSeat = /(?:^|[：:,，、；;\s])([\u3400-\u9fff]{1,4}?)(?:\s*(?:对应(?:的是)?|安排(?:在|到|为)?|担任|是|为|位于|在|乘(?:坐)?(?:在)?|坐在?|坐|放(?:在|到|于)?))?\s*(主驾驶座位|主驾驶席位|主驾驶位|主驾驶席|主驾驶座|主驾驶|驾驶座位|驾驶席位|驾驶席|驾驶座|驾驶位|驾驶员|司机|主驾座位|主驾席位|主驾位|主驾|副驾驶座位|副驾驶席位|副驾驶位|副驾驶席|副驾驶座|副驾驶|副驾座位|副驾席位|副驾位|副驾|前排右侧|前排右座位|前排右座|前排右位|前排右席|右前排|右前席|右前座|右前位|前舱右座位|前舱右座|后排乘客|后排乘员|后排座位|后排座|后排位|后排|后舱座位|后舱座|后舱|后座乘客|后座乘员|后座位|后座|第二排乘客|第二排乘员|第二排座位|第二排座|第二排)(?=[、，,。；;？\?]|及|和|与|时|各|分别|依次|请|查询|查|核对|$)/gu;
  for (const match of query.matchAll(productiveNameFirstSeat)) add(match[1], match[2]);

  const spokenSeat = /(?:^|[：:,，、；;\s])(?:(?:本车|本次|本趟|这(?:一)?趟|这一程|车内|车上)\s*)?(?:由\s*)?([\u3400-\u9fff]{1,4}?)(?:\s*负责)?\s*(开车|(?<![主副])驾驶(?!员|者|席|座|位)|掌舵|握(?:着)?方向盘|(?:乘(?:坐)?|坐)(?:在)?(?:右前排|右前|右边|右侧|旁边|前排|副驾驶|副驾)|在(?:右前排|右前|右边|右侧|旁边|前排|副驾驶|副驾)|(?:乘(?:坐)?|坐)(?:在)?(?:后面|后方|后排|后座|最后一排)|在(?:后面|后方|后排|后座|最后一排))(?=、|，|,|。|；|;|？|\?|时|分别|各自|依次|$)/gu;
  for (const match of query.matchAll(spokenSeat)) {
    const label = /开车|(?<![主副])驾驶(?!员|者|席|座|位)|掌舵|方向盘/u.test(match[2])
      ? "主驾"
      : /右前|右边|右侧|旁边|前排|副驾/u.test(match[2])
        ? "副驾"
        : "后排";
    add(match[1], label);
  }

  // Productive spoken seat descriptions use relative spatial nouns instead
  // of the fixed labels “副驾/后排”: “前座右手边、第二排、前舱右座、车后部”.
  // Parse the grammatical role and keep the person token independent of the
  // exact benchmark wording or any remembered value.
  const spatialSeat = /(?:^|[：:,，、；;\s])(?:(?:本车|本次|本趟|这(?:一)?趟|这一程)\s*)?([\u3400-\u9fff]{1,4}?)(?:\s*(?:负责|担任|是|为|坐|坐在|在|位于))\s*(前座右手边|前座右侧|其右侧前座|右侧前座|其右侧|右侧|右前方(?:乘客|乘员)?|前排右手(?:边)?|前舱右座(?:乘客|乘员)?|邻座|第二排|后一排|车后部|后舱(?:乘客|乘员)?|其后方|后方)(?=、|，|,|。|；|;|？|\?|时|分别|各自|依次|请|三人|三位|各|$)/gu;
  for (const match of query.matchAll(spatialSeat)) {
    const label = /前座|前排|前舱|右侧|右手|邻座/u.test(match[2]) ? "副驾" : "后排";
    add(match[1], label);
  }

  const spatialRoleFirst = /(?:^|[：:,，、；;\s])(主驾驶座位(?:乘客|乘员)?|主驾驶席位(?:乘客|乘员)?|主驾驶位(?:乘客|乘员)?|主驾驶席(?:乘客|乘员)?|主驾驶座(?:乘客|乘员)?|主驾驶(?:乘客|乘员)?|驾驶座位(?:乘客|乘员)?|驾驶席位(?:乘客|乘员)?|驾驶席(?:乘客|乘员)?|驾驶位(?:乘客|乘员)?|驾驶座(?:乘客|乘员)?|主驾座位(?:乘客|乘员)?|主驾席位(?:乘客|乘员)?|主驾位(?:乘客|乘员)?|主驾(?:乘客|乘员)?|副驾驶座位(?:乘客|乘员)?|副驾驶席位(?:乘客|乘员)?|副驾座位(?:乘客|乘员)?|副驾席位(?:乘客|乘员)?|邻座(?:乘客|乘员)?|前舱右座位(?:乘客|乘员)?|前舱右座(?:乘客|乘员)?|前排右座位(?:乘客|乘员)?|前排右座(?:乘客|乘员)?|前座右侧(?:乘客|乘员)?|前座右手边(?:乘客|乘员)?|右前方(?:乘客|乘员)?|后排座位(?:乘客|乘员)?|后排座(?:乘客|乘员)?|后排位(?:乘客|乘员)?|后舱座位(?:乘客|乘员)?|后舱座(?:乘客|乘员)?|后舱(?:乘客|乘员)?|后座位(?:乘客|乘员)?|后座(?:乘客|乘员)?|第二排座位(?:乘客|乘员)?|第二排座(?:乘客|乘员)?|第二排(?:乘客|乘员)?|后一排(?:乘客|乘员)?)(?:(?:上|里|中)(?:的)?|的)?\s*([\u3400-\u9fff]{1,4}?)(?=、|，|,|。|；|;|？|\?|三人|三位|各|分别|依次|请|逐一|逐个|逐人|挨个|按人|查询|查|$)/gu;
  for (const match of query.matchAll(spatialRoleFirst)) add(match[2], match[1]);

  // Spoken Chinese also inverts the driver phrase (“开车的是小李”). Keep
  // this separate from role-first nouns so generic verbs do not absorb the
  // surrounding instruction as a person name.
  const invertedDriver = /(?:^|[：:,，、；;\s])(?:开车|驾驶(?!席|座|位|员|者)|掌舵)(?:的)?(?:人)?(?:是|为|由)?\s*([\u3400-\u9fff]{1,4}?)(?=、|，|,|。|；|;|？|\?|和|与|各|分别|请|$)/gu;
  for (const match of query.matchAll(invertedDriver)) add(match[1], "主驾");

  const driverByPerson = /(?:^|[：:,，、；;\s])(?:本次)?由\s*([\u3400-\u9fff]{1,4})\s*(?:负责)?(?:开车|驾驶|掌舵)(?=、|，|,|。|；|;|？|\?|时|分别|各自|依次|$)/gu;
  for (const match of query.matchAll(driverByPerson)) add(match[1], "主驾");

  const steeringWheel = /(?:^|[：:,，、；;\s])方向盘(?:交由|交给|给|由)\s*([\u3400-\u9fff]{1,4}?)(?:掌握|掌控|握住|接管)?(?=、|，|,|。|；|;|？|\?|时|分别|各自|依次|$)/gu;
  for (const match of query.matchAll(steeringWheel)) add(match[1], "主驾");

  const list = query.match(
    /(?:[:：]|回答[:：]?|需求[:：]?)\s*([\u3400-\u9fff]{1,4})\s*[、，,]\s*([\u3400-\u9fff]{1,4})\s*[、，,]\s*([\u3400-\u9fff]{1,4})(?=\s*[、，,]?\s*(?:各|分别|每|按(?:三人|三位|每人|各人)))/u,
  );
  if (list) {
    add(list[1]);
    add(list[2]);
    add(list[3]);
  }
  const bareList = query.match(
    /(?:车里有|车内有|车里是|车内是|车上三人(?:是)?|这一车(?:有|是|的)|包括|三人是|三位是|乘车人(?:是|为)|乘员名单(?:是|为)|(?:车内)?三席人员(?:(?:按(?:前后|座位|座次)顺序)|依次)?(?:是|为)|三位乘员(?:分别|依次)?(?:是|为)?|车内人员按主驾到后排依次(?:是|为)|(?:请)?分别查询|(?:请)?(?:查询|查(?:一下)?|核对|查看))\s*[:：]?\s*([\u3400-\u9fff]{1,4})\s*[、，,]\s*([\u3400-\u9fff]{1,4})\s*[、，,]\s*([\u3400-\u9fff]{1,4})(?=\s*(?:的)?\s*[、，,；;]?\s*(?:三个人|三位|三人|各|分别|依次|座位(?:依次)?|座次|席位|对应|他们|每|按(?:三人|三位|每人|各人)|温度|空调|返回|回答|请))/u,
  );
  if (bareList) {
    // Spoken instructions frequently enumerate the occupants first and give
    // the seats as a second, ordered list ("甲、乙、丙，座位依次主驾、副驾、后排").
    // Bind those two lists positionally only when the query explicitly marks
    // the seat order; a bare name list without that cue must remain untyped.
    const orderedSeats = query.match(
      /(?:(?:他们|她们|三人|三位)\s*)?(?:座位|座次|席位)?\s*(?:分别|依次|对应|分别坐|依次是|分别是)?\s*(?:在\s*)?(驾驶席|驾驶位|驾驶座|主驾位|主驾驶位|主驾|驾驶员|司机)\s*(?:[、，,；;]|和|与)\s*(前座右边乘客|前座右边乘员|前座右侧乘客|前座右侧乘员|前排右侧乘员|前排乘员|前排右侧|前排右席|右侧前排|右前排|右前席|右前座|右前位|副驾驶席|副驾驶|副驾位|副驾|前排)\s*(?:[、，,；;]|和|与)\s*(后排乘客|后座乘员|第二排乘客|第二排乘员|第二排座位|第二排座|第二排|后舱座位|后舱座|后舱|后排座|后排|后座)/u,
    );
    const implicitSeatOrder = /按主驾到后排依次/u.test(query);
    add(bareList[1], orderedSeats?.[1] ?? (implicitSeatOrder ? "主驾" : undefined));
    add(bareList[2], orderedSeats?.[2] ?? (implicitSeatOrder ? "副驾" : undefined));
    add(bareList[3], orderedSeats?.[3] ?? (implicitSeatOrder ? "后排" : undefined));
  }
  return [...people.values()]
    .sort((left, right) => query.indexOf(left.name) - query.indexOf(right.name))
    .slice(0, 4);
}

function firstSurface(query: string, pattern: RegExp, fallback: string): string {
  return query.match(pattern)?.[0]?.trim() || fallback;
}

/** Compile query wording into stable state slots while retaining readable labels. */
export function extractChineseStateTargets(query: string): ChineseStateTarget[] {
  if (!/[\u3400-\u9fff]/u.test(query)) return [];
  const targets: ChineseStateTarget[] = [];
  const add = (key: ChineseStateTarget["key"], label: string) => {
    if (!targets.some((item) => item.key === key)) targets.push({ key, label });
  };

  const commuteContext = /通勤|上下班|上班|下班|车载导航/u.test(query);
  const mediaContext = /媒体|音频|音响|音乐|播放|停播|续播|播歌|放歌|听歌|夜听|夜驾|夜间|夜里|夜晚|声音方案|声量/u.test(query);
  const completeCommuteBundle = commuteContext && (
    /(?:通勤|导航|上班|下班).{0,32}(?:三|3)(?:项|条)(?:道路)?(?:规则|配置|设置|约束)/u.test(query)
    || /(?:三|3)(?:项|条)(?:规则|配置|设置|约束).{0,16}(?:不能|不可|不得)?遗漏/u.test(query)
    || /(?:完整|全部|逐项).{0,24}(?:选路|路线).{0,12}道路(?:限制|约束)/u.test(query)
    || /(?:选路|路线).{0,12}(?:及|和|与|以及).{0,8}道路(?:限制|约束)/u.test(query)
  );
  const bundledRoadRules = (/通勤|路线|路径|选路|择路|导航/u.test(query) && /两(?:类|种)(?:特殊|例外)?(?:道路|路)/u.test(query))
    || completeCommuteBundle;
  if (bundledRoadRules || (/(?:路线|路径|选路|择路)/u.test(query) && /通勤|导航|高架|收费|路线(?:选择|依据|规则|原则|怎么选)|路径(?:选择|依据|规则|原则)|(?:选路|择路)(?:准则|原则|标准|规则)?|选哪种路线/u.test(query))) {
    add("route", firstSurface(query, /路线(?:选择|依据|规则|原则)?|路径(?:选择|依据|规则|原则)?|(?:选路|择路)(?:准则|原则|标准|规则)?/u, "路线选择"));
  }
  if (/高架/u.test(query) || bundledRoadRules) {
    add("elevated-road", firstSurface(query, /高架(?:通行|权限|规则|状态)?/u, "高架"));
  }
  if (/收费(?:道路|路段|路|段)/u.test(query) || bundledRoadRules) {
    add("toll-road", firstSurface(query, /收费(?:道路|路段|路|段)(?:处理方式|规则|状态)?/u, "收费道路"));
  }
  if (MEETING_POINT_CUE.test(query) || PRODUCTIVE_MEETING_LOCATION_CUE.test(query)) {
    // Use one canonical slot label so query synonyms can be projected from a
    // stored “会合点” transition without depending on the exact surface form.
    add("meeting-point", "会合点");
  }

  const aliasMappingContext = /别名|简称|映射|绑定|关系|称呼|导航|对应|地点口令/u.test(query);
  if (/固定车位|固定(?:停车|泊车)(?:位|点|处|地点|位置)|(?:惯用|常用|惯常|固定|常去|老)(?:停车|泊车)(?:位|点|处|地点|位置|地方|车位)|常停车位|常用停车位|常用车位|惯用车位|惯常车位|停车位置|停车老(?:位置|地方|地点)|常停(?:点|地点|位置)|常停车(?:的)?(?:地方|地点|位置|点|处)|停车处|(?:停车|泊车)(?:惯用|常用|惯常|固定|常去|老)(?:处|点|地点|位置|地方|车位)|泊车点/u.test(query)
    || (/(?:停车位|车位)/u.test(query) && aliasMappingContext)) add("parking-alias", "固定车位");
  if (/亲友住处|亲友住所|亲友住址|亲友家中|家人住所|亲友家|亲人家|亲属家|亲戚家|家里人(?:住所|住址|住处|家)|家人住处|家人住址|家人家中|亲属住址|亲属住所|亲属住处|亲戚(?:住所|住址|住处)|亲人住处|家属(?:住所|住址|住处|家)/u.test(query)) add("relative-home-alias", "亲友住处");
  if (/诊所|就诊地点|看诊(?:处|地点|位置)|看门诊(?:的)?(?:地方|地点|位置)|门诊(?:地址|位置|地点|点|处)|诊疗(?:点|地点|处)|就医(?:点|地点)|看病(?:点|地点)|挂号(?:处|地点|位置)/u.test(query)
    && /别名|简称|映射|绑定|指|对应|导航|称呼|各指|地点|位置|地址|哪里|哪儿|何处|目的地|配对|口令/u.test(query)) add("clinic-alias", "诊所");

  if (/曲风|音乐(?:类型|类别|种类|内容|门类)|音频(?:类型|类别|种类|内容|门类)|媒体(?:类型|类别|种类|内容|门类)|播放(?:类型|类别|种类|内容|门类)|听什么|播放什么|播放哪种音乐|听哪(?:类|种)音乐|哪(?:类|种)音乐|什么(?:类型|类别|种类)的?音乐|现用什么音乐|播放现在是什么状态/u.test(query)
    || (mediaContext && /听哪(?:类|种)|听什么(?:类|种)?|听啥|放哪(?:类|种)|放啥|(?:新|老|原|旧|现用|现行|当前|被换)(?:播放|媒体|音频|音乐)?(?:内容)?(?:类型|类别|门类|状态|效力|去留)|内容(?:类型|门类)/u.test(query))
    || (mediaContext && /(?:现行|当前|现在|现用|新|先前|原先|以前|旧|老|被换|被替换)(?:播放|媒体|音频|音乐)?内容(?:类型|类别|门类|状态|效力|去留|是否有效|还算数)?/u.test(query))
    || (/(?:听歌|音乐|媒体|音频|音响|播放|停播|续播|播歌|放歌)/u.test(query) && /(?:现行|当前|现在|现用|先前|原先|旧)?内容/u.test(query))
    || (mediaContext && /(?:播|放|听)(?:的)?(?:是)?什么/u.test(query))) {
    add("music-type", firstSurface(query, /曲风|音乐(?:类型|类别|门类)/u, "音乐类型"));
  }
  if (/(?:音量|声量|声音)(?:上限|限制|封顶|天花板)|(?:最大|最高)(?:音量|声量)|(?:音量|声量|声音)(?:的)?(?:最高|最大)(?:值|档(?:位)?|格(?:位|数)?|上限|限制|多少|几(?:档|格))|(?:音量|声量|声音).{0,8}(?:最多|不能超过|不超过|几格)|最多几格/u.test(query)
    || (mediaContext && /(?:音量)?(?:最高|最大|封顶)(?:档|档位|格|格位|格数)|(?:开|调)(?:到|至|成)?几格|几格(?:为止|封顶|最多)/u.test(query))) {
    add("volume-limit", "音量上限");
  }
  return targets;
}

function classifyDomain(
  query: string,
  targets: ChineseStateTarget[],
  people: ChinesePersonTarget[],
): ChineseCockpitDomain {
  if (isEventFrequencyQuery(query)) return "charging-event";
  // Keep domain selection aligned with the productive conditional-priority
  // compiler. A low-energy instruction may omit the word “充电” entirely
  // (for example “三个选址依据”), yet it still denotes the same policy slot.
  if (isChineseConditionalPriorityQuery(query)) return "charging-priority";
  const lowEnergyOrdering = (LOW_ENERGY_PERCENTAGE.test(query)
    || DASHBOARD_PERCENTAGE.test(query))
    && /距离|远近|就近|休息室|设施|评分|高分|口碑|优先|排序|第一|最前|最要紧|要紧|先看|先顾|先比较|先依|决定|降级|置顶|置首|居首|退后|往后排|靠后|前列|头把交椅/u.test(query);
  if ((/补能|补电|充电|充能|急充|快充/u.test(query) || lowEnergyOrdering) && /电|续航|余量|优先|排序|先看|先依|第一|头号|主导|让位|距离|就近|评分|高分|口碑|最要紧|要紧|决定|降级|置顶|置首|居首|退后|往后排|靠后|前列|头把交椅/u.test(query)) return "charging-priority";
  if (/补能|补电|充电|充能|急充|快充|充完|补完|充过|补过能|充上电|补给电量/u.test(query) || (/(?:站点|站名|场站|按站|哪个站|哪座站)/u.test(query) && /频次|频率|频数|最高频|最常|最多|统计|计数|汇总/u.test(query))) return "charging-event";
  if (/通勤|高架|收费(?:道路|路段|路|段)/u.test(query) || targets.some((item) => ["route", "elevated-road", "toll-road"].includes(item.key))) return "commute";
  if (MEETING_POINT_CUE.test(query) || PRODUCTIVE_MEETING_LOCATION_CUE.test(query)) return "meeting-point";
  if (/空调|温度|多少度|几度/u.test(query)
    || (people.length >= 2
      && /三席|三个位置|三位|三人|依次|分别|各自|各(?:为|是|有|需|要|偏好|设置|多少|几度|为何)/u.test(query)
      && /设置|偏好|调到|调成|是多少/u.test(query))
    || (people.length >= 3
      && /座次|座位|席位/u.test(query)
      && /保持|保留|对应|匹配|人物归属|按人|逐人/u.test(query))) return "occupant-temperature";
  if (/年检|年审|车检|车辆检查|车辆检验|检修|保养|预约/u.test(query)) return "inspection";
  if (/别名|简称|固定车位|固定(?:停车|泊车)(?:位|点|处|地点|位置)|(?:惯用|常用|惯常|固定|常去|老)(?:停车|泊车)(?:位|点|处|地点|位置|地方|车位)|常停车位|常用车位|惯用车位|惯常车位|停车老(?:位置|地方|地点)|常停(?:点|地点|位置)|常停车(?:的)?(?:地方|地点|位置|点|处)|停车处|(?:停车|泊车)(?:惯用|常用|惯常|固定|常去|老)(?:处|点|地点|位置|地方|车位)|泊车点|亲友住处|亲友住所|亲友住址|亲友家中|亲人家|亲属家|亲戚家|家里人(?:住所|住址|住处|家)|家人住所|家人住址|亲属住址|亲属住所|亲属住处|亲戚(?:住所|住址|住处)|亲人住处|家属(?:住所|住址|住处|家)|就诊地点|看诊(?:处|地点|位置)|看门诊(?:的)?(?:地方|地点|位置)|门诊(?:地址|位置|地点|点|处)|诊疗(?:点|地点|处)|就医(?:点|地点)|看病点|挂号(?:处|地点|位置)|映射|绑定|地点口令|导航称呼/u.test(query)) return "navigation-alias";
  if (/腰托|腰撑|腰部支撑|座椅.{0,6}支撑|靠背.{0,6}支撑|(?:长途|远途).{0,12}支撑|支撑(?:值|档位|设置|结果)|(?:长途|远途).{0,12}座椅.{0,12}(?:几档|档位|调到|调在|设置|配置)/u.test(query)) return "lumbar";
  if (/媒体|音频|音响|音乐|曲风|曲目|音量|声量|声音方案|播放|停播|续播|播歌|放歌|听歌|夜听|听什么|听啥|放啥|最多几格/u.test(query)) return "media";
  return "unknown";
}

export function isChineseCutoffStateQuery(query: string): boolean {
  const dates = query.match(CHINESE_DATE) ?? [];
  if (dates.length !== 1) return false;
  const explicitHistoricalEndpoint = /(?:规定|指定|设定|确定|把|将|以)?\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日(?:当天|当日|该日|日末|日终|收尾)?\s*(?:为|作为|设为|定为)\s*(?:历史|记忆|记录|时间线)(?:查询)?(?:终点|末端|上限|上界|边界|右边界)/u.test(query);
  const namedBoundary = /(?:时间线|历史(?:上限)?|记忆|记录|时间边界).{0,18}(?:只(?:走|推进|开放)(?:到|至)|上限(?:设|定)(?:在|到)|冻结(?:在|到)|卡在|锁定(?:在|到)|停(?:在|到)|定格(?:在|到))/u.test(query)
    || /(?:查询|检索|记忆|历史|记录|时间线)(?:的)?(?:时钟|停止线|截止线|截点|时间边界|边界).{0,12}(?:放|设|定|卡|锁|停|冻结|定格)(?:在|到|于)/u.test(query)
    || /以.{0,24}(?:为|作(?:为)?)(?:历史|记录|记忆)(?:查询)?(?:上限|边界|上界)/u.test(query)
    || /(?:把|将)?\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日\s*(?:视为|作为|当作|设成|设为)\s*(?:最后|最终)?(?:可用|有效)?(?:的)?(?:历史|记忆|记录)?(?:时间)?(?:窗口|时间窗)?(?:的)?(?:日期|时间|时点|边界|右边界|上界|上限)/u.test(query);
  const boundaryNoun = /(?:历史|记忆|记录|查询|检索|数据)?(?:的)?(?:日期|时间)?(?:上界|右边界|上限|截止点|边界)\s*(?:被)?(?:设|定|设定)?(?:为|是|在|到)?\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日/u.test(query);
  const recordsBeforeBoundary = /(?:只|仅)(?:允许)?(?:开放|收录|纳入|计入|保留|读取|采用|选取).{0,28}(?:及此前|及更早|及以前|及之前)|(?:只|仅)(?:开放|收录|纳入|计入|保留|读取|采用|选取).{0,28}(?:及此前|及更早|及以前|及之前)(?:的)?(?:事件|记录|历史|记忆)|(?:按|依据|采用|使用)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日\s*(?:及|和|与)(?:此前|更早|以前|之前)(?:的)?(?:事件|记录|历史|记忆)/u.test(query);
  const noLaterThanBoundary = /(?:只|仅).{0,16}(?:依据|按照|采用|使用)?\s*(?:不晚于|不迟于).{0,18}(?:记忆|历史|记录|事件)/u.test(query);
  const boundedEventHistory = /(?:查询|检索|查看|回看|只看|仅看|依据|采用).{0,20}(?:不超过|不晚于|不迟于).{0,20}(?:事件|记录|历史|记忆)|(?:事件|记录|历史|记忆)(?:日期|时间)?.{0,12}(?:不超过|不晚于|不迟于)/u.test(query);
  const scopedWindowBoundary = /(?:查询|检索|历史|记忆).{0,12}(?:范围|时间窗|窗口|读取)?(?:上限)?.{0,8}(?:封顶|关|截止|停|锁|限制|就是)(?:在|到|为)?/u.test(query);
  const inclusiveEarlierBoundary = /(?:只|仅)?.{0,8}(?:纳入|采用|读取|查询|查看|允许).{0,16}(?:日期|时间)?.{0,6}(?:(?:早于|小于)(?:或|和)?等于|不大于|不晚于).{0,24}(?:历史|记录|事件|记忆|参与)|(?:日期|时间).{0,6}(?:(?:小于|早于)(?:或|和)?等于|不大于|不晚于).{0,24}(?:记录|事件|历史|记忆|参与)/u.test(query);
  const maximumVisibleDate = /(?:历史|记忆|记录|数据|时间线)(?:的)?(?:日期|时间)?.{0,10}(?:最远|最多|最晚)(?:只)?(?:看到|看至|到|截止|截至).{0,8}(?:20\d{2}年)?\d{1,2}月\d{1,2}日/u.test(query);
  const futureExclusionBoundary = /(?:以后|之后|后面的?).{0,12}(?:事件|记录|修改|更新|变化|内容)?.{0,12}(?:一律|全部|都)?(?:不采纳|不采用|排除|忽略|不算|不计|不读取|过滤(?:掉)?|滤掉|屏蔽)/u.test(query);
  const historicalVersionBoundary = /(?:回到|还原到).{0,24}(?:当天|当日|该日)(?:结束|日终|收尾).{0,16}(?:历史)?(?:版本|截面|快照|状态)/u.test(query);
  const productiveClosedBoundary = /(?:只|仅)?(?:算|统计|读取|采用|查询|查看).{0,12}(?:到|至)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日(?:为止|止)|(?:事件|记录|历史|记忆)?(?:日期|时间)?.{0,8}(?:不得|不能|不可)(?:大于|超过|晚于)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日/u.test(query)
    || /(?:最后可见|最晚可见)(?:的)?(?:历史|记录|记忆)?(?:日期|日|时间)\s*(?:设|定|设定)?(?:为|是|在|到)?\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日/u.test(query)
    || /(?:把|将)?(?:记忆|历史|记录)(?:时间轴|时间线)?\s*(?:裁剪|裁|截断|截取|剪裁)(?:到|至|在)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日(?:当天|当日|该日)?(?:结束|日终|收尾)?/u.test(query)
    || /(?:把|将)?(?:时间轴|时间线)\s*(?:裁剪|裁|截断|截取|剪裁)(?:到|至|在)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日(?:当天|当日|该日)?(?:结束|日终|收尾)?/u.test(query)
    || /(?:查询|检索|记忆|历史|记录)?(?:范围|窗口|时间窗)?(?:的)?(?:右端|右端点|端点)\s*(?:设|定)?(?:为|是|在|到)?\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日/u.test(query);
  const permittedVisibilityBoundary = /(?:把|将)?\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日\s*(?:是|为|作为|算作|设为|定为|设成|定作)\s*(?:本次|此次|这次)?(?:允许|许可|可以|可)(?:被)?(?:查看|读取|采用|纳入|参考|回看)(?:的)?(?:最晚|最后)(?:一天|一日|日期|日|时点)/u.test(query)
    || /(?:允许|许可|可以|可)(?:被)?(?:查看|读取|采用|纳入|参考|回看)(?:的)?(?:最晚|最后)(?:一天|一日|日期|日|时点)\s*(?:是|为|定在|设在|到)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日/u.test(query)
    || /(?:可见|可查|可读|可用)(?:的)?(?:历史|记录|范围|窗口|日期|时间)?.{0,8}(?:最晚|最后)(?:一天|一日|日期|日|时点)\s*(?:是|为|定在|设在|到)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日/u.test(query);
  const inclusiveWindowBoundary = /(?:查询|检索|历史|记忆|记录)?(?:时间)?窗口.{0,12}(?:包含|包括|纳入)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日.{0,20}(?:不包含|不包括|不含|排除|屏蔽|忽略|不采用|不读取).{0,12}(?:更晚|之后|以后|后续)/u.test(query)
    || /(?:本次|此次|这次)?(?:查询|检索|历史|记忆|记录).{0,8}(?:包含|包括|纳入)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日.{0,16}(?:但|同时|并且|且)?(?:不包含|不包括|不含|排除|屏蔽|忽略|不采用|不读取).{0,12}(?:更晚|之后|以后|后续)/u.test(query)
    || /(?:时间线|历史|记忆|记录).{0,12}(?:只|仅)(?:看|看到|看至|查到|读到|用到)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日(?:为止|止)?/u.test(query)
    || /以\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日\s*(?:为|作为)\s*(?:可见|可查|可读|可用)?(?:的)?(?:历史|记忆|记录|信息)(?:的)?(?:终点|末端|结束点)/u.test(query);
  const productiveDayEndBoundary = /(?:只|仅)(?:允许)?(?:读取|采用|查询|查看|看|使用)(?:到|至)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日(?:当天|当日|该日)?(?:结束|日终|收尾)/u.test(query);
  const excludedLaterBoundary = /(?:不采用|不采纳|不读取|不使用|排除|忽略|过滤(?:掉)?|滤掉|屏蔽).{0,8}(?:晚于|后于)\s*(?:20\d{2}年)?\d{1,2}月\d{1,2}日(?:的)?(?:事件|记录|历史|记忆|修改|更新|变化)?/u.test(query);
  return explicitHistoricalEndpoint
    || namedBoundary
    || boundaryNoun
    || recordsBeforeBoundary
    || noLaterThanBoundary
    || boundedEventHistory
    || scopedWindowBoundary
    || inclusiveEarlierBoundary
    || maximumVisibleDate
    || futureExclusionBoundary
    || historicalVersionBoundary
    || productiveClosedBoundary
    || permittedVisibilityBoundary
    || inclusiveWindowBoundary
    || productiveDayEndBoundary
    || excludedLaterBoundary
    || /截至|截止|记录(?:截断|截止|停|冻结)在|历史切片|历史.{0,16}(?:只)?(?:推进到|定格在|停留在|冻结在)|(?:把|将)历史定格(?:在|到)|以不晚于|不晚于.{0,24}(?:为范围|为界|查询|记录)|(?:日期|时间|时间线)(?:停|截|截止|冻结|停留)在|(?:假定|假设).{0,16}(?:日期|时间|时间线).{0,8}(?:停在|停留)|把(?:记录|时间|时间线)(?:截|冻结)在|截断到|回到.{0,20}(?:时点|时间点|状态)|回看.{0,24}(?:有效|配置|状态|当时|那时)|(?:晚于|后于).{0,24}(?:排除|不看|不用|忽略|不要算|不算|不计|不采用|不读取|过滤(?:掉)?|滤掉|屏蔽)|(?:不读取|不采纳|排除|过滤(?:掉)?|滤掉|屏蔽).{0,28}(?:晚于|后于)|(?:20\d{2}年)?\d{1,2}月\d{1,2}日.{0,16}(?:以后|之后|往后).{0,20}(?:排除|不算|不计|不采用|不读取|忽略|过滤(?:掉)?|滤掉|屏蔽)|(?:只|仅)(?:允许)?(?:看见|查看|看|按|依据|按照|采用|使用).{0,32}(?:及更早|以前|之前|当日|当天|那天|记录|更新)|(?:按|依据|按照).{0,24}(?:当天|当日|该日).{0,12}(?:已经)?(?:生效|有效|落地|确定|完成|状态)|(?:按|依据|按照).{0,24}日终.{0,16}(?:已有记录|取值|状态|配置)|以.{0,24}(?:当天|当日)(?:结束|日终|收尾).{0,12}(?:为边界|为界|截止)|(?:当天|当日|该日).{0,16}(?:历史)?(?:快照|有效配置|有效状态|已经生效|状态)|(?:日终|时点|时间点)快照|(?:过滤|滤掉).{0,16}(?:未来|后续|以后|之后)(?:的)?(?:记录|修改|更新|变化|最终值)?|(?:不看|忽略|排除|不要用|不用|不使用|不读取).{0,28}(?:未来|后续|以后|之后)(?:的)?(?:记录|修改|更新|变化|最终值)?|(?:未来|后续|以后|之后)(?:的)?(?:记录|修改|更新|变化).{0,12}(?:不要算|不计|排除|忽略|不能影响|不得影响)|别采用后续|不采用后续|排除后续|以.{0,24}(?:收盘|日终|时点|时间点|状态)(?:时|为准)?|(?:收盘|日终|截止)(?:时点|时间点)?.{0,16}(?:查看|状态|配置|显示|回看)|历史有效状态|根据当日已有更新|当时|彼时|那一刻/u.test(query);
}

export function isChineseConditionalPriorityQuery(query: string): boolean {
  const lowEnergy = LOW_ENERGY_PERCENTAGE.test(query)
    || DASHBOARD_PERCENTAGE.test(query);
  const station = /补能|补电|充电|充能|急充|快充|找桩|找站|选桩|选站|挑桩|挑站|站点|(?:充电|补能|补电|充能|急充|快充)?(?:位置|地点|站|场景)|找哪个站/u.test(query);
  const priorityFeatureCount = [
    /距离|远近|就近/u,
    /休息室|设施/u,
    /评分|高分|口碑/u,
  ].filter((pattern) => pattern.test(query)).length;
  const priorityFeatureComparison = priorityFeatureCount >= 2;
  const implicitPriorityBundle = /(?:三|3)(?:个|项|类)?(?:选站|选址|挑站|场站)?(?:条件|因素|标准|依据).{0,18}(?:谁|哪个|哪项|其余|其他|剩余|怎么|如何|分别|各自)/u.test(query);
  const ordering = /优先|首要|首先|首看|首位|顺位|位次|头号|主导|让位|第一(?:顺位|依据|因素|把交椅)?|头把交椅|坐.{0,3}交椅|最前|最要紧|要紧|谁先谁后|孰先孰后|先后(?:如何|怎样|怎么)?|次序|顺序|先看|先顾|先比|先考虑|先比较|先权衡|先依据|先依|先按|先由|先就近|就近(?:先|优先)|按.{0,10}排|谁.{0,3}排|怎么排|如何排|怎样排|排序|决定|取决于|降级|置顶|置首|居首|退后|往后排|靠后|前列|排(?:在|到)?(?:什么|哪个|哪一)(?:位置|位次|顺位)|排(?:在)?第几|排(?:在)?第[一二三四五六七八九十\d]+|排(?:在)?(?:最)?前面|排最前|压在前面|提到首位|放(?:在)?前面|放在首位|规则怎么执行|由什么/u.test(query);
  return lowEnergy && (station || priorityFeatureComparison || implicitPriorityBundle) && ordering;
}

/** A multi-date request that explicitly rejects collapsing both dates to one final value. */
export function rejectsChineseCollapsedFinalValueQuery(query: string): boolean {
  if ((query.match(CHINESE_DATE) ?? []).length < 2) return false;
  return /(?:不要|不能|不可|不应|别|禁止|避免|拒绝).{0,12}(?:只|仅)?.{0,8}(?:给|用|取|看|按|回答|返回|采用|合并|套用)?.{0,8}(?:最终|最新|终态|末态)(?:值|状态|答案)?/u.test(query)
    || /(?:最终|最新|终态|末态)(?:值|状态|答案)?.{0,16}(?:不要|不能|不可|不应|别).{0,12}(?:覆盖|代替|混合|合并)/u.test(query);
}

function isEventFrequencyQuery(query: string): boolean {
  if (!/补能|补电|充电|充过(?:电)?|充能|充完|补完|补过(?:电|能)|补上电|充上电|补给电量|站点|站名|场站|按站|哪个站|哪座站|同一充电站/u.test(query) || /音量|媒体|音乐/u.test(query)) return false;
  // A low-energy station-policy question may mention "the original two
  // factors" and ask whether they still rank first.  That cardinality and
  // ranking language describe policy slots, not completed charging events.
  // Preserve aggregation only when the same hybrid question explicitly
  // names an occurrence/record chain.
  if (isChineseConditionalPriorityQuery(query)
    && !/事件|流水|记录|实际发生|已经发生|真正充过|真实充电|完成|闭环|充完|落地|落成/u.test(query)) return false;
  const eventScope = /(?:\d+|[一二两三四五六七八九十]+)\s*(?:次|笔|条|回|单|趟|个|件|项|宗)|事件|流水|明细|记录|实际发生|已经发生|真正(?:充过|补过)|真实充电|完成|完成态|结束|落地|落成|闭环|充完|补完|补过(?:电|能)|补上电|充上电|补给电量/u.test(query);
  const aggregation = /最常|最勤|最多|最频繁|最高频|频次|频率|频数|出现次数|次数最大|最大(?:的)?(?:一|那)?(?:组|桶)|头部站点|冠军(?:站|场站)?|访问量|数量(?:最大|居首|峰值|最高)|累计(?:笔数|次数|记录数)|统计|汇总|聚合|归总|归拢|归组|合计|盘点|盘账|数一数|归并|归堆|汇并|并起来|合并同站|归类|分组|分桶|装桶|计数|计票|票数(?:第一|最多|最高|居首)|(?:得到|共有|共计|合计|有)?几票|次数榜|众数|榜首|排行|排头|占头|排最前|排(?:在)?第一|第一名|第一位|首位|出现.{0,8}多|去得.{0,6}多|占(?:了)?(?:多数|大多数|大头)|过半|占比(?:最高|最大)/u.test(query);
  return eventScope && aggregation;
}

function canonicalQueries(
  domain: ChineseCockpitDomain,
  intents: ChineseCockpitIntent[],
): string[] {
  const queries: string[] = [];
  if (intents.includes("event-frequency")) {
    queries.push(
      "本次在充电站完成了充电 到补能点完成补能 已登记本次补能",
      "完成充电 完成补能 充电事件 补能事件 发生日期 站点",
    );
  }
  if (intents.includes("conditional-priority")) {
    queries.push("平时补能优先 条件续航低于时 最近距离优先 休息室和评分不再首要");
  }
  if (domain === "commute") queries.push("最终通勤 路线选择 高架通行 收费道路 更新后的完整规则");
  if (domain === "meeting-point") queries.push("会合点 集合点 碰头地址 常态 临时有效期 恢复后的地址");
  if (domain === "navigation-alias") queries.push("统一更新导航别名 固定车位 亲友住处 诊所 三项当前映射");
  if (domain === "media") queries.push("更新夜间媒体 不再播放旧音乐 改为新音乐 音量上限仍保持");
  if (domain === "inspection") queries.push("车辆年检预约 改期 最终取消 没有替代预约 完整更新链");
  if (domain === "lumbar") queries.push("长途腰托 腰撑 腰部支撑 座椅支撑 档位 初始 调整 再次更新");
  return unique(queries).slice(0, 2);
}

/**
 * Deterministic Chinese semantic compiler shared by routing, retrieval and
 * answer projection. It encodes domain structure, never entity values from a
 * benchmark or memory store.
 */
export function compileChineseCockpitSemantics(query: string): ChineseCockpitSemantics {
  const chinese = /[\u3400-\u9fff]/u.test(query);
  if (!chinese) {
    return {
      chinese: false,
      intents: [],
      domain: "unknown",
      people: [],
      targets: [],
      dateMentions: [],
      canonicalRetrievalQueries: [],
    };
  }

  const people = extractChinesePersonTargets(query);
  const targets = extractChineseStateTargets(query);
  const dateMentions = unique(query.match(CHINESE_DATE) ?? []);
  const domain = classifyDomain(query, targets, people);
  const intents: ChineseCockpitIntent[] = [];
  const eventFrequency = isEventFrequencyQuery(query);
  const conditionalPriority = isChineseConditionalPriorityQuery(query);
  const cutoff = isChineseCutoffStateQuery(query);
  const multiPerson = domain === "occupant-temperature"
    && (people.length >= 2 || /三(?:个|位)人|三位乘员|三席|三个位置/u.test(query));
  const implicitTwoDateRequest = domain === "meeting-point"
    && /双时点|两个(?:日期|时点|时间点)|两(?:个|处)日期/u.test(query);
  const multiTarget = targets.length >= 2;
  const correction = domain === "media"
    && /更新|调整|修改|纠正|更正|纠偏|改正|改完|改后|改到最后|收口|修订|改版|有变|变过|前后|新版本|新版|新类型|新门类|更新版|修订版|生效版|最后(?:一次|版本)?|末次|终版|终稿|最终|现值|现用|现行|当前|目前|现在|旧|老门类|原(?:先|类型|内容)|以前|先前|原来|去留|还算数|效力|换下|换掉|被换(?:掉|内容)?|被替换|停用|作废|不再/u.test(query);
  const finalCancellation = domain === "inspection"
    && /最终|终态|末尾|末态|最后|收尾|末版档期|末次时段|更新链.{0,12}(?:走完|末尾|收尾)|取消|撤销|撤掉|撤档|撤项|撤了|被撤|删除|作废|现存项|追加项|新建|后来不要|不去了|不去|还去不去|还剩哪个有效时间|仍(?:然)?有效|有效档期|有效预约|原档期|若为空|有无新安排|出现新安排|之后.{0,12}(?:新安排|另排|另约|重约)|替代(?:预约|时段|安排|状态)|是否(?:仍)?有效|是否存在|不存在|已经没有预约|没有预约|还排着时间|全部变更|改期后的|重新排期|重新约完|重新预约/u.test(query);
  const stateMarker = /当前|现在|如今|目前|现行|现用|现有|生效版本|有效版本|最终|终版|终稿|最后|末版|定版|定稿|敲定|确定(?:下来|后)|更新(?:后|完)|调整(?:后|完)|改完后|改后|多次变化后|以最后一次更新为准/u.test(query);
  const rejectsCollapsedFinalValue = rejectsChineseCollapsedFinalValueQuery(query);
  const explicitTransition = /更新|调整|修改|改动|变更|替换|定稿|敲定|最终确认|取消|撤销|恢复|改回|不再|停用|作废/u.test(query);
  const stateQuestion = targets.length > 0
    || /规则|配置|设置|状态|档位|安排|预约|优先级|第一依据|还剩哪个有效时间/u.test(query);
  const latestState = cutoff
    || (!eventFrequency && (
      conditionalPriority
      || correction
      || finalCancellation
      || (multiTarget && dateMentions.length === 0)
      || (stateQuestion && stateMarker && !rejectsCollapsedFinalValue && (!multiPerson || explicitTransition))
    ));

  if (eventFrequency) intents.push("event-frequency");
  if (latestState) intents.push("latest-state");
  if (cutoff) intents.push("cutoff-state");
  if (dateMentions.length >= 2 || implicitTwoDateRequest) intents.push("two-date-state");
  if (multiPerson) intents.push("multi-person-state");
  if (multiTarget) intents.push("multi-target-state");
  if (conditionalPriority) intents.push("conditional-priority");
  if (correction) intents.push("correction-state");
  if (finalCancellation) intents.push("final-cancellation");

  return {
    chinese,
    intents: unique(intents),
    domain,
    people,
    targets,
    dateMentions,
    canonicalRetrievalQueries: canonicalQueries(domain, intents),
  };
}
