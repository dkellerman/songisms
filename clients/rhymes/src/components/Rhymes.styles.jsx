import styled from 'styled-components';

export const DEFAULT_GRID_HEIGHT = '50vh';

export const StyledRhymes = styled.article`
  fieldset {
    display: flex;
    align-items: center;
    margin: 20px 0 12px 0;

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
`;

export const ColumnLayout = styled.ul`
  list-style: none;
  padding-left: 0;
  display: flex;
  flex-flow: column wrap;
  max-width: 700px;
  max-height: ${DEFAULT_GRID_HEIGHT};
  gap: 7px;
`;

export const RhymeItem = styled.li`
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
`;
