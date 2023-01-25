export function getTeamColor(team: number, alpha: number): string {
  const rgb = team === 0 ? [34, 139, 230] : [240, 62, 62];
  return `rgba(${rgb.join(', ')}, ${alpha})`;
}
