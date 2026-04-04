# Data endpoints used in this project

## Official APIs used by default

### CDC Socrata API
- Provisional Drug Overdose Death Counts:
  - Dataset page: https://data.cdc.gov/National-Center-for-Health-Statistics/VSRR-Provisional-Drug-Overdose-Death-Counts/xkb8-kh2a
  - JSON endpoint: https://data.cdc.gov/resource/xkb8-kh2a.json
- County-Level Provisional Drug Overdose Death Counts:
  - Dataset page: https://data.cdc.gov/National-Center-for-Health-Statistics/VSRR-Provisional-County-Level-Drug-Overdose-Death-/gb4e-yj24
  - JSON endpoint: https://data.cdc.gov/resource/gb4e-yj24.json
- Specific Drugs Overdose Counts:
  - Dataset page: https://data.cdc.gov/National-Center-for-Health-Statistics/Provisional-drug-overdose-death-counts-for-specifi/8hzs-zshh
  - JSON endpoint: https://data.cdc.gov/resource/8hzs-zshh.json

### U.S. Census API
- Developer page: https://www.census.gov/data/developers.html
- API user guide: https://www.census.gov/data/developers/guidance/api-user-guide.html
- Example ACS endpoint pattern:
  - https://api.census.gov/data/2023/acs/acs5?get=NAME,B01003_001E&for=state:20&key=YOUR_KEY

## Optional API
### Reddit API (OAuth required)
- OAuth docs: https://www.reddit.com/dev/api/oauth/
- API wiki: https://www.reddit.com/r/reddit.com/wiki/api/

## Caveat
Google Trends has an alpha API with limited tester access, so this starter kit does not depend on it by default.
