/** Strict representation helpers for already-extracted cockpit event times. */

const ZONED_ISO_TIMESTAMP = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?(Z|([+-])(\d{2}):(\d{2}))$/u;

const CONTROLLED_EVENT_TIME_SLOTS = new Set([
  "navigation.departure_time",
  "navigation.arrival_time",
  "navigation.pickup_time",
  "schedule.appointment_time",
  "reminder.reminder_time",
]);

export function isControlledCockpitEventTimeSlot(
  domainValue: unknown,
  slotValue: unknown,
): boolean {
  return typeof domainValue === "string"
    && typeof slotValue === "string"
    && CONTROLLED_EVENT_TIME_SLOTS.has(`${domainValue.trim()}.${slotValue.trim()}`);
}

/**
 * Parse only a real calendar timestamp with an explicit UTC offset.
 *
 * JavaScript's permissive Date parser rolls invalid dates such as February 30
 * into March.  Construction-time projection and alias proof must fail closed
 * instead, so every calendar and clock component is validated before the UTC
 * instant is returned.  Fractional precision beyond milliseconds is accepted
 * but compared at the persistence layer's millisecond resolution.
 */
export function strictZonedIsoInstant(value: unknown): number | undefined {
  if (typeof value !== "string") return undefined;
  const match = ZONED_ISO_TIMESTAMP.exec(value.trim());
  if (!match) return undefined;
  const [
    , yearText, monthText, dayText, hourText, minuteText,
    secondText = "0", fractionText = "", zone, sign, offsetHourText, offsetMinuteText,
  ] = match;
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const hour = Number(hourText);
  const minute = Number(minuteText);
  const second = Number(secondText);
  const millisecond = Number(fractionText.slice(0, 3).padEnd(3, "0"));
  if (year < 1 || month < 1 || month > 12 || day < 1
    || hour > 23 || minute > 59 || second > 59) return undefined;
  const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const daysInMonth = [31, leapYear ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1];
  if (day > daysInMonth) return undefined;

  let offsetMinutes = 0;
  if (zone !== "Z") {
    const offsetHour = Number(offsetHourText);
    const offsetMinute = Number(offsetMinuteText);
    // The civil-time offset range is bounded at UTC+/-14:00.
    if (offsetHour > 14 || offsetMinute > 59 || (offsetHour === 14 && offsetMinute !== 0)) {
      return undefined;
    }
    offsetMinutes = (sign === "-" ? -1 : 1) * (offsetHour * 60 + offsetMinute);
  }

  const local = new Date(0);
  local.setUTCFullYear(year, month - 1, day);
  local.setUTCHours(hour, minute, second, millisecond);
  if (local.getUTCFullYear() !== year
    || local.getUTCMonth() !== month - 1
    || local.getUTCDate() !== day
    || local.getUTCHours() !== hour
    || local.getUTCMinutes() !== minute
    || local.getUTCSeconds() !== second) return undefined;
  return local.getTime() - offsetMinutes * 60_000;
}
