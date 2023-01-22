export function getTeamColor(team: number, alpha: number, minimalTheme: boolean): string {
  let rgb: [number, number, number];
  if (minimalTheme) {
    rgb = team === 0 ? [34, 139, 230] : [240, 62, 62];
  } else {
    rgb = team === 0 ? [224, 65, 40] : [0, 112, 81];
  }

  return `rgba(${rgb.join(', ')}, ${alpha})`;
}

export function getTeamColorMantine(team: number, minimalTheme: boolean): string {
  if (minimalTheme) {
    return team === 0 ? 'blue' : 'red';
  } else {
    return team === 0 ? 'red' : 'green';
  }
}
