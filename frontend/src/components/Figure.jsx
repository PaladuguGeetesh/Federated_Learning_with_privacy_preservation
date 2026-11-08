import { useEffect, useState } from 'react'

// A diagram that opens full-screen when clicked (the diagrams contain small text).
export default function Figure({ src, alt, caption }) {
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (!open) return
    const on = (e) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [open])

  return (
    <figure className="figure">
      <button className="figure-btn" onClick={() => setOpen(true)} aria-label={`Enlarge: ${alt}`}>
        <img src={src} alt={alt} loading="lazy" />
        <span className="figure-zoom">Click to enlarge</span>
      </button>
      {caption && <figcaption>{caption}</figcaption>}
      {open && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label={alt} onClick={() => setOpen(false)}>
          <img src={src} alt={alt} />
          <button className="lightbox-close" aria-label="Close">×</button>
        </div>
      )}
    </figure>
  )
}
