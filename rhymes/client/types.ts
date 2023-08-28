export interface Rhyme {
  text: string;
  frequency?: number;
  type: 'rhyme'|'rhyme-l2'|'suggestion';
  isTop?: boolean;
}

export type Completion = string;
