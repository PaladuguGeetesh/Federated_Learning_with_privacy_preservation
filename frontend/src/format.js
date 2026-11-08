export const int = (n) => (n == null ? '–' : Number(n).toLocaleString('en-US'))
export const pct = (x, d = 2) => (x == null ? '–' : `${(x * 100).toFixed(d)}%`)
export const fixed = (x, d = 4) => (x == null ? '–' : Number(x).toFixed(d))

// Cell values of the data set: whole numbers in full, decimals to 4 places.
export const cell = (v) => {
  if (v == null) return '–'
  if (Number.isInteger(v)) return v.toLocaleString('en-US')
  const a = Math.abs(v)
  if (a !== 0 && (a < 0.0001 || a >= 1e9)) return v.toExponential(3)
  return v.toLocaleString('en-US', { maximumFractionDigits: 4 })
}

// Number of parameters of the MLP (input -> hidden layers -> 1 logit).
export const annParams = (input, hidden) => {
  let prev = input
  let total = 0
  for (const h of [...hidden, 1]) {
    total += prev * h + h
    prev = h
  }
  return total
}
