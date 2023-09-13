export interface Rhyme {
  text: string;
  type: 'rhyme'|'rhyme-l2'|'suggestion';
  frequency?: number;
  vote?: string;
  source?: string;
  score?: number;
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

export interface VoteRequest {
  anchor: string;
  alt1: string;
  label: string;
  voter_uid: string;
  remove?: string;
}

export type RLHF = {
  anchor: string;
  alt1: string;
  alt2: string;
};

export type RLHFResponse = {
  hits: RLHF[];
};
