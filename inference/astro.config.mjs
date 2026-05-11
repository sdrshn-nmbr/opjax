import { defineConfig, passthroughImageService } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'http://localhost:4322',
  image: {
    service: passthroughImageService(),
  },
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { strict: false, throwOnError: false }]],
    syntaxHighlight: 'shiki',
    shikiConfig: {
      theme: 'rose-pine-moon',
      wrap: true,
    },
  },
  build: {
    inlineStylesheets: 'auto',
  },
  vite: {
    ssr: {
      noExternal: ['katex'],
    },
    server: {
      allowedHosts: ['.tail625050.ts.net'],
    },
  },
});
