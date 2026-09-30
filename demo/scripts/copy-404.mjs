import { copyFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
copyFileSync(resolve(root, 'dist/index.html'), resolve(root, 'dist/404.html'))
console.log('postbuild: dist/404.html written')
