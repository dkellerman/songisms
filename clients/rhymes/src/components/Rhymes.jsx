import { useCallback, useEffect, useRef, useState, useMemo } from 'react';
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
  const [page, setPage] = useState(1);
  const { rhymes, loading, abort, hasNextPage } = useRhymes(q, searchType, page, PAGE_SIZE);

  const inputRef = useRef();
  const suggestRef = useRef();

  const search = useCallback((newQ, newSearchType) => {
    if (!(newQ || '').trim()) return;

    setQ(newQ);
    setSearchType(newSearchType);
    setPage(1);
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
      rhyme: rhymes?.filter(r => ['rhyme', 'top'].includes(r.type)).length || 0,
      ror: rhymes?.filter(r => ['rhyme-l2'].includes(r.type)).length || 0,
      sug: rhymes?.filter(r => ['suggestion'].includes(r.type)).length || 0,
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
          placeholder="Search for rhymes in songs..."
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
            {!q && <label>Top {counts.rhyme} rhymes</label>}
            {q && (
              <label>
                {!counts.sug && `${ct2str(counts.rhyme, 'rhyme')} found`}
                {counts.ror > 0 && `, ${ct2str(counts.ror, 'rhyme-of-rhyme', 'rhymes-of-rhymes')}`}
                {counts.sug > 0 && `${ct2str(counts.sug, 'suggestion')}`}
              </label>
            )}

            <ColumnLayout>
              {rhymes.map(r => (
                <RhymeItem key={r.ngram}>
                  <span className={`hit ${r.type}`} onClick={() => search(r.ngram, searchType)}>
                    {r.ngram}
                  </span>{' '}
                  {!!r.frequency && <span className="freq">({r.frequency})</span>}
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
