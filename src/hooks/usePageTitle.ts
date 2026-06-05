import { useEffect } from 'react'

/** Sets the document title for the page; restores nothing on unmount since
 * every page sets its own. */
export function usePageTitle(title: string) {
  useEffect(() => {
    document.title = title
  }, [title])
}
