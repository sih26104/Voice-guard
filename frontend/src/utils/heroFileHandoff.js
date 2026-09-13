/**
 * In-memory handoff for a file chosen on the homepage hero card.
 *
 * React Router route state must be JSON-serializable, so a File object
 * cannot travel through navigate('/dashboard', { state: ... }). This tiny
 * module holds the File in memory for exactly one dashboard consumption —
 * no persistence, no duplicates. It carries no analysis data; the file is
 * still uploaded by the dashboard's real /api/analyze flow.
 */

let handoffFile = null;

/** Store the file picked on the homepage (replaces any previous handoff). */
export function setHandoffFile(file) {
  handoffFile = file;
}

/** Consume the handoff (returns the file once, then clears it). */
export function takeHandoffFile() {
  const file = handoffFile;
  handoffFile = null;
  return file;
}
