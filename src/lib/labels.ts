// The upstream labeling pipeline capped cluster labels at 50 characters, so
// a third of the skill names end mid-word ("...In Quantu"). Until the labels
// are regenerated, append an ellipsis so the cut reads as deliberate
// truncation rather than a rendering bug.
export function displaySkillLabel(label: string): string {
  if (label.length >= 50 && !/[.!?)…]$/.test(label)) return `${label}…`
  return label
}
