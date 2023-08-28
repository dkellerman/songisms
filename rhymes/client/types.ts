export interface Rhyme {
  text: string;
  frequency?: number;
  type: 'rhyme'|'rhyme-l2'|'suggestion';
}

export interface Completion {
  text: string;
};

export interface RhymesResponse {
  isTop: boolean;
  hits: Rhyme[];
}


export interface CompletionsResponse {
  hits: Completion[];
}
