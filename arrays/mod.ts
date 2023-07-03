export function range(start: number, end: number, step: number): number[] {
  const a = [];
  let b = start;
  while (b < end) {
    a.push(b);
    b += step;
  }
  return a;
}

export function arrayEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}
