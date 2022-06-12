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
  const [page, setPage] = useState(1);
  const { rhymes, loading, abort, hasNextPage } = useRhymes(q, page, PAGE_SIZE);
  const inputRef = useRef();

  const search = useCallback((newQ) => {
    const qVal = (newQ || '').trim();
    if (!qVal) return;

    setQ(qVal);
    setPage(1);

    track('engagement', 'search', qVal);

    const routerQuery = { ...router.query, q: qVal };
    return router.push({ query: routerQuery });
  }, [router]);

  const debouncedSearch = useCallback(
    debounce(e => search(e.target.value), SEARCH_DEBOUNCE),
    [search],
  );

  const onInput = useCallback(async e => {
    abort?.();  // cancel any current requests
    return debouncedSearch(e);
  },[debouncedSearch, abort]);

  useEffect(() => {
    if (!inputRef.current) return;
    inputRef.current.value = q ?? '';
  }, [q]);

  useEffect(() => {
    setQ(router.query?.q ?? '');
  }, [router.query]);

  const counts = useMemo(
    () => ({
      rhyme: rhymes?.filter(r => r.type === 'rhyme').length || 0,
      l2: rhymes?.filter(r => r.type === 'rhyme-l2').length || 0,
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
      </fieldset>

      <output>
        {loading && <label>Searching...</label>}

        {!loading && !!rhymes && (
          <>
            {!q && <label>Top {counts.rhyme} rhymes</label>}

            {!!q && [
              `${ct2str(counts.rhyme, 'rhyme')} found`,
              counts.l2 > 0 && ct2str(counts.l2, 'rhyme-of-rhyme', 'rhymes-of-rhymes'),
              counts.sug > 0 && ct2str(counts.sug, 'suggestion'),
            ].filter(Boolean).join(', ')}

            <ColumnLayout>
              {rhymes.map(r => (
                <RhymeItem key={r.ngram}>
                  <span className={`hit ${r.type}`} onClick={() => search(r.ngram)}>
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
                  track('engagement', 'more', q);
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
