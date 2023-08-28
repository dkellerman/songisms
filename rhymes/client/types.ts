export interface Rhyme {
  text: string;
  frequency?: number;
  type: 'rhyme'|'rhyme-l2'|'suggestion';
}

export interface Completion {
  text: string;
};

export interface RhymesRequest {
  q: string;
  limit?: number;
}

export interface RhymesResponse {
  isTop: boolean;
  hits: Rhyme[];
}


export interface CompletionsRequest {
  q: string;
  limit?: number;
}

export interface CompletionsResponse {
  hits: Completion[];
}
