import styled, { css } from 'styled-components';

export const StyledRhymes = styled.article`
  fieldset {
    display: flex;
    flex-flow: row wrap;
    align-items: center;
    width: 100%;
    margin: 20px 0 12px 0;
    padding-right: 10px;
    gap: 20px;

    input[type='text'] {
      border-radius: 0;
      width: 50vw;
      min-width: 200px;
      max-width: 500px;
      &::-webkit-search-cancel-button {
        -webkit-appearance: searchfield-cancel-button;
      }
    }
    input[type='checkbox'] {
      zoom: 1.5;
      display: inline-block;
      margin-right: 5px;
    }
    label {
      font-size: large;
      position: relative;
      top: -3px;
    }
  }

  output label {
    font-size: large;
  }
`;

const gap = 20;
const colWidth = (cols) => css`calc((100% - ${(cols - 1) * gap}px) / ${cols})`;

export const ColumnLayout = styled.ul`
  list-style: none;
  padding-left: 0;
  display: flex;
  flex-flow: row wrap;
  max-width: 768px;
  gap: ${gap}px;
`;

export const RhymeItem = styled.li`
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

  @media screen and (max-width: 374px) {
    width: ${colWidth(1)};
  }
  @media screen and (min-width: 375px) and (max-width: 479px) {
    width: ${colWidth(2)};
  }
  @media screen and (min-width: 480px) {
    width: ${colWidth(3)};
  }
`;
