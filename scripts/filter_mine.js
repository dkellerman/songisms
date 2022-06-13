#!/usr/bin/env node

const fs = require('fs');
const data = fs.readFileSync('./data/mine_raw.txt', 'utf8');

// prettier-ignore
const lines = uniq(
  data
    .split('\n')
    .map((l) =>
      l
        .replace(/\[[^\]]*\]/g, '')
        .replace(/\.+/g, '')
        .replace(/[[\]]/g, '')
        .replace(/^\((.*)\)$/, '$1')
        .replace(/\bcos\b/gi, '\'cause')
        .replace(/^\s*-\s*/g, '')
        .replace(/''/g, '\'')
        .replace(/^[\d.\s]*/g, '')
        .replace(/…/g, '')
        .replace(/\s+/g, ' ')
        .replace(/’/g, "'")
        .trim()
        .toLowerCase()
        .split('/')
        .map((x) => x.trim())
    )
    .flat()
    .filter(
      (line) =>
        line &&
        line.indexOf(';') === -1 &&
        line.indexOf('_') === -1 &&
        line.indexOf('|') === -1 &&
        (line.match(/\s/g)?.length || 0) > 2 &&
        (line.match(/\s/g)?.length || 0) < 10 &&
        line.match(/[A-Za-z]+/) &&
        !line.match(/m7/gi)
    )
);

function uniq(arr) {
  return arr.filter((value, index, self) => self.indexOf(value) === index);
}

fs.writeFileSync('./data/mine.txt', lines.join('\n'));
