import React, { useCallback, useEffect, useRef, useState, useMemo } from 'react';
import debounce from 'lodash/debounce';
import { isMobile } from 'react-device-detect';
import { useRouter } from 'next/router';
import { useRhymes } from '../hooks/useRhymes';
import { ColumnLayout, RhymeItem, StyledRhymes } from './Rhymes.styles';

const SEARCH_DEBOUNCE = isMobile ? 1000 : 500;
const PAGE_SIZE = 50;

export default function Rhymes() {
  const router = useRouter();
  const [q, setQ] = useState(router.query ? router.query.q : '');
  const [searchType, setSearchType] = useState(router.query ? router.query.t : 'rhyme');

  const showTop = !q;
  const showSuggestions = searchType === 'suggest';
  const showRhymes = searchType !== 'suggest';

  const [page, setPage] = useState(1);
  const [n, setN] = useState(showSuggestions ? [3, 3] : undefined);
  const { rhymes, loading, abort, hasNextPage } = useRhymes(q, searchType, page, PAGE_SIZE, n);

  const inputRef = useRef();
  const suggestRef = useRef();

  const search = useCallback((newQ, newSearchType) => {
    if (!(newQ || '').trim() && (searchType === newSearchType))
      return;

    setQ(newQ);
    setSearchType(newSearchType);
    setPage(1);

    if (newSearchType !== 'suggest') setN(undefined);
    else if (newQ) setN([1, 3]);
    else setN([3, 3]);

    track('engagement', `${newSearchType || 'rhyme'}`, newQ);

    const routerQuery = { ...router.query, q: newQ, t: newSearchType };
    if (!newQ?.trim()) delete routerQuery.q;
    if (newSearchType !== 'suggest') delete routerQuery.t;
    return router.push({ query: routerQuery });
  }, [router]);

  const debouncedSearch = useCallback(
    debounce(e => search(e.target.value, searchType), SEARCH_DEBOUNCE),
    [search, searchType],
  );

  const onInput = useCallback(async e => {
    abort?.();  // cancel any current requests
    return debouncedSearch(e);
  },[debouncedSearch, abort]);

  const onSetSearchType = useCallback(
    async e => {
      search(q, e.target.checked ? 'suggest' : 'rhyme');
    },
    [q, search],
  );

  useEffect(() => {
    if (!inputRef.current || !suggestRef.current) return;
    inputRef.current.value = q ?? '';
    suggestRef.current.checked = searchType === 'suggest';
  }, [q, searchType]);

  useEffect(() => {
    setQ(router.query?.q || '');
    setSearchType(router.query?.t || 'rhyme');
  }, [router.query]);

  const counts = useMemo(
    () => ({
      rhyme: rhymes?.filter(r => r.type === 'rhyme').length || 0,
      ror: rhymes?.filter(r => r.type === 'rhyme-l2').length || 0,
      sug: rhymes?.filter(r => r.type === 'suggestion').length || 0,
    }),
    [rhymes],
  );

  return (
    <StyledRhymes>
      <fieldset>
        <input
          ref={inputRef}
          type="text"
          onChange={onInput}
          defaultValue={q}
          placeholder="Find rhymes in songs..."
        />
        <div className="suggestions">
          <input type="checkbox" ref={suggestRef} onInput={onSetSearchType} />
          <label>Suggestions</label>
        </div>
      </fieldset>

      <output>
        {loading && <label>Searching...</label>}

        {!loading && !!rhymes && (
          <>
            {showTop && showRhymes && counts.rhyme > 0 && <label>Top {counts.rhyme} rhymes</label>}
            {showTop && showSuggestions && counts.sug > 0 && <label>Top {counts.sug} suggestions</label>}
            {!showTop && (
              <label>
                {showRhymes && `${ct2str(counts.rhyme, 'rhyme')} found`}
                {showRhymes && counts.ror > 0 && `, ${ct2str(counts.ror, 'rhyme-of-rhyme', 'rhymes-of-rhymes')}`}
                {showSuggestions && `${ct2str(counts.sug, 'suggestion')} found`}
              </label>
            )}
            {showSuggestions && (
              <>
                <span className="word-links">[&nbsp; Words: &nbsp;
                  <span onClick={() => setN(null)}>all</span>
                  <span onClick={() => setN([1, 1])}>1</span>
                  <span onClick={() => setN([2, 2])}>2</span>
                  <span onClick={() => setN([3, 3])}>3</span>
                  <span onClick={() => setN([3, null])}>3+</span>
                ]&nbsp;</span>
              </>
            )}

            <ColumnLayout>
              {rhymes.map(r => (
                <RhymeItem key={r.ngram}>
                  <span className={`hit ${r.type}`} onClick={() => search(r.ngram, searchType)}>
                    {r.ngram}
                  </span>{' '}
                  {!!r.frequency && r.type === 'rhyme' && <span className="freq">({r.frequency})</span>}
                </RhymeItem>
              ))}
            </ColumnLayout>

            {hasNextPage && (
              <button
                className="more compact"
                onClick={() => {
                  track('engagement', `more_${searchType}`, q);
                  setPage(page + 1);
                }}
              >
                More...
              </button>
            )}
          </>
        )}
      </output>
    </StyledRhymes>
  );
}

function track(category, action, label) {
  if (window.gtag) {
    window.gtag('event', action, {
      event_category: category,
      event_label: label,
    });
  }
}

function ct2str(ct, singularWord, pluralWord) {
  const plWord = pluralWord ?? `${singularWord}s`;
  if (ct === 0) return `No ${plWord}`;
  if (ct === 1) return `1 ${singularWord}`;
  return `${ct} ${plWord}`;
}
