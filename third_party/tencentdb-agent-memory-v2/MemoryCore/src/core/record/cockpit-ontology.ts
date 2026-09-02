/**
 * Stable, schema-level cockpit ontology helpers.
 *
 * These mappings describe ownership of contract slots; they do not contain
 * dataset values, entities, dates, people, or utterance-specific rules.
 */

const CONTROLLED_SLOT_OWNERS = new Map<string, readonly string[]>([
  ...[
    "origin", "destination", "waypoint", "route_constraint", "departure_time",
    "arrival_time", "pickup_time", "pickup_person", "guidance_volume_limit",
  ].map((slot) => [slot, ["navigation"]] as const),
  ...[
    "category_constraint", "location_constraint", "rating_constraint",
    "price_constraint", "duration_constraint", "feature_constraint",
    "ranking_policy", "release_period_constraint",
  ].map((slot) => [slot, ["selection"]] as const),
  ...["appointment_time", "appointment_content"].map((slot) => [slot, ["schedule"]] as const),
  ...["reminder_time", "reminder_content"].map((slot) => [slot, ["reminder"]] as const),
  ["broadcast_policy", ["notification"]],
  ...["temperature", "fan_speed"].map((slot) => [slot, ["climate"]] as const),
  ...["media_title", "playlist", "playback_status"].map((slot) => [slot, ["media"]] as const),
  ...["contact", "message_content", "call_status"].map((slot) => [slot, ["communication"]] as const),
  ...["position", "heating_level", "ventilation_level"].map((slot) => [slot, ["seat"]] as const),
  ...["window_state", "door_state", "charging_status"].map((slot) => [slot, ["vehicle_control"]] as const),
  ["status", [
    "navigation", "selection", "schedule", "reminder", "notification",
    "climate", "media", "communication", "seat", "vehicle_control",
  ]],
]);

const SLOT_ALIASES = new Map<string, string>([
  ["default_destination", "destination"],
  ["score_constraint", "rating_constraint"],
  ["minimum_rating", "rating_constraint"],
  ["min_rating", "rating_constraint"],
  ["budget_constraint", "price_constraint"],
  ["cost_constraint", "price_constraint"],
  ["fare_constraint", "price_constraint"],
  ["visit_duration", "duration_constraint"],
  ["play_duration", "duration_constraint"],
  ["proximity_constraint", "location_constraint"],
  ["amenity_constraint", "feature_constraint"],
  ["amenities_constraint", "feature_constraint"],
  ["sort_rule", "ranking_policy"],
  ["ranking_rule", "ranking_policy"],
  ["decade_constraint", "release_period_constraint"],
  ["volume_limit", "guidance_volume_limit"],
  ["navigation_volume_limit", "guidance_volume_limit"],
]);

const CONSTRAINT_TARGET_ALIASES = new Map<string, string>([
  ["ticket", "ticket"],
  ["ticket_price", "ticket"],
  ["门票", "ticket"],
  ["票价", "ticket"],
  ["per_capita", "per_capita"],
  ["per_person", "per_capita"],
  ["人均", "per_capita"],
  ["人均消费", "per_capita"],
  ["room", "room"],
  ["room_rate", "room"],
  ["房价", "room"],
  ["住宿", "room"],
  ["generic", "generic"],
  ["通用", "generic"],
]);

function canonicalToken(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value
    .normalize("NFKC")
    .trim()
    .toLocaleLowerCase()
    .replace(/[\s-]+/gu, "_");
  return normalized || undefined;
}

export function canonicalCockpitDomain(value: unknown): string | undefined {
  return canonicalToken(value);
}

export function canonicalCockpitSlot(value: unknown): string | undefined {
  const slot = canonicalToken(value);
  return slot ? SLOT_ALIASES.get(slot) ?? slot : undefined;
}

export function canonicalCockpitConstraintTarget(value: unknown): string | undefined {
  const target = canonicalToken(value);
  return target ? CONSTRAINT_TARGET_ALIASES.get(target) ?? target : undefined;
}

export function controlledCockpitSlotOwners(slotValue: unknown): readonly string[] | undefined {
  const slot = canonicalCockpitSlot(slotValue);
  return slot ? CONTROLLED_SLOT_OWNERS.get(slot) : undefined;
}

export function canonicalControlledCockpitDomain(
  domainValue: unknown,
  slotValue: unknown,
): string | undefined {
  const domain = canonicalCockpitDomain(domainValue);
  const owners = controlledCockpitSlotOwners(slotValue);
  if (owners?.length === 1) return owners[0];
  return domain;
}

/**
 * Stable persistence/reconciliation class for a cockpit fact.
 *
 * `scene_name` is a model-authored grouping label and can legitimately drift
 * between a session title, a free-text scene, and a domain token.  A fact in
 * the controlled cockpit ontology instead belongs to the canonical owner of
 * its slot.  Uncontrolled memories retain their original scene verbatim so
 * this helper does not redefine TencentDB's generic scene semantics.
 */
export function canonicalCockpitSceneClass(
  sceneValue: unknown,
  domainValue: unknown,
  slotValue: unknown,
): string | undefined {
  const scene = typeof sceneValue === "string" && sceneValue.trim().length > 0
    ? sceneValue
    : undefined;
  const slot = canonicalCockpitSlot(slotValue);
  const owners = controlledCockpitSlotOwners(slot);
  const domain = canonicalControlledCockpitDomain(domainValue, slot);
  if (slot && owners && domain && owners.includes(domain)) return domain;
  return scene;
}

export function isValidControlledCockpitOntology(
  domainValue: unknown,
  slotValue: unknown,
): boolean {
  const domain = canonicalCockpitDomain(domainValue);
  const owners = controlledCockpitSlotOwners(slotValue);
  return !domain || !owners || owners.includes(domain);
}
