## Pricing

Pricing is based on the number of tokens used, or other metrics based on the model type. For tool-specific models, like search and computer use, there’s a fee per tool call. See details in the [pricing page](/api/docs/pricing).

### Text tokens

| Metric | Price | Unit |
| --- | ---: | --- |
| Input | $0.1 | 1M tokens |
| Cached input | $0.01 | 1M tokens |
| Cache writes | $0.125 | 1M tokens |
| Output | $0.5 | 1M tokens |

- Cached input tokens are priced at 10% of the uncached input token rate.
- Cache writes are billed at 1.25x the uncached input token rate.
- Prompts with more than 272K input tokens are priced at 2x input and cache rates and 1.5x output for the full request.
- Regional processing adds a 10% premium where available. EU data residency is available only with Standard processing.
- Batch and Flex are priced at 50% of Standard rates. Fast mode is priced at 2x the applicable rates.
