import { useCallback, useEffect, useRef, useState, useLayoutEffect, useMemo } from 'react';
import styled from 'styled-components';
import debounce from 'lodash/debounce';
import { isMobile } from 'react-device-detect';
import { useRouter } from 'next/router';
import { useRhymes } from '../hooks/useRhymes';

const DEFAULT_HEIGHT = '50vh';
const SEARCH_DEBOUNCE = isMobile ? 1000 : 500;

const StyledRhymes = styled.article`
  fieldset {
    display: flex;
    align-items: center;
    margin: 10px 0 12px 0;

    input[type='search'] {
      width: 65vw;
      min-width: 180px;
      max-width: 500px;
      &::-webkit-search-cancel-button {
        -webkit-appearance: searchfield-cancel-button;
      }
    }
    input[type='checkbox'] {
      margin: 0 7px 0 20px;
      zoom: 1.5;
    }
    label {
      font-size: large;
      position: relative;
      top: 3px;
    }
  }

  output label {
    font-size: large;
  }

  ul {
    list-style: none;
    padding-left: 0;
    display: flex;
    flex-flow: column wrap;
    max-width: 700px;
    max-height: ${DEFAULT_HEIGHT};
    gap: 7px;
    li {
      white-space: nowrap;
      text-indent: 0;
      font-size: larger;
      &:before {
        display: none;
      }
      .freq {
        font-size: medium;
        color: #666;
      }
      .hit {
        text-decoration: underline;
        color: blue;
        cursor: pointer;
        &.rhyme-l2 {
          opacity: 0.6;
        }
        &.suggestion {
          opacity: 0.6;
        }
      }
    }
  }

  button.more {
    padding: 10px;
    margin-top: 15px;
    font-size: medium;
  }
`;

export default function Rhymes() {
  const router = useRouter();
  const [q, setQ] = useState(router.query ? router.query.q : '');
  const [searchType, setSearchType] = useState(router.query ? router.query.t : 'rhyme');
  const [limit, setLimit] = useState(50);
  const { rhymes, loading } = useRhymes(q, searchType, limit);
  const inputRef = useRef();
  const suggestRef = useRef();
  const listRef = useRef();

  const search = useCallback((newQ, newSearchType) => {
    setQ(newQ);
    setSearchType(newSearchType);
    setLimit(50);
    listRef.current.style.maxHeight = DEFAULT_HEIGHT;

    const newQuery = { ...router.query, q: newQ, t: newSearchType };
    if (!newQ?.trim()) delete newQuery.q;
    if (newSearchType !== 'suggest') delete newQuery.t;
    return router.push({ query: newQuery });
  }, []);

  const onInput = useCallback(
    debounce(async e => {
      search(e.target.value?.trim(), searchType);
    }, SEARCH_DEBOUNCE),
    [searchType],
  );

  const onSetSearchType = useCallback(
    async e => {
      search(q, e.target.checked ? 'suggest' : 'rhyme');
    },
    [q],
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

  useLayoutEffect(() => {
    resize();
  }, [rhymes]);

  useEffect(() => {
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);

  function resize() {
    if (!listRef.current) return;
    const { scrollWidth, clientWidth, clientHeight } = listRef.current;
    if (scrollWidth > clientWidth) {
      listRef.current.style.maxHeight = `${clientHeight * 2}px`;
    }
  }

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
          type="search"
          onChange={onInput}
          defaultValue={q}
          placeholder="Search for rhymes in songs..."
        />
        <input type="checkbox" ref={suggestRef} onInput={onSetSearchType} />
        <label>Suggestions (beta)</label>
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

            <ul ref={listRef}>
              {rhymes.map(r => (
                <li key={r.ngram}>
                  <span className={`hit ${r.type}`} onClick={() => search(r.ngram, searchType)}>
                    {r.ngram}
                  </span>{' '}
                  {!!r.frequency && <span className="freq">({r.frequency})</span>}
                </li>
              ))}
            </ul>

            {limit < 200 && rhymes.length >= limit && (
              <button className="more" onClick={() => setLimit(limit + 50)}>
                More...
              </button>
            )}
          </>
        )}
      </output>
    </StyledRhymes>
  );
}

function ct2str(ct, singularWord, pluralWord) {
  const plWord = pluralWord ?? `${singularWord}s`;
  if (ct === 0) return `No ${plWord}`;
  if (ct === 1) return `1 ${singularWord}`;
  return `${ct} ${plWord}`;
}
