import { defineCollection, z } from 'astro:content';

const chapters = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    chapter: z.number(),
    pageStart: z.number().optional(),
    pageEnd: z.number().optional(),
  }),
});

export const collections = { chapters };
