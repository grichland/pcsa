'use strict';

// usage: node validate.cjs <showdown root> <format> <teams.json>
// teams.json: a JSON list of Showdown export strings. Prints a JSON list of {index, ok, problems}.
const path = require('node:path');
const fs = require('node:fs');

const [root, format, teamsPath] = process.argv.slice(2);
const {Teams, TeamValidator} = require(path.join(root, 'dist', 'sim'));
const validator = TeamValidator.get(format);
const teams = JSON.parse(fs.readFileSync(teamsPath, 'utf8'));
const results = teams.map((text, index) => {
  const team = Teams.import(text);
  if (!team) return {index, ok: false, problems: ['could not import team']};
  const problems = validator.validateTeam(team);
  return {index, ok: !problems, problems: problems || []};
});
process.stdout.write(JSON.stringify(results));
