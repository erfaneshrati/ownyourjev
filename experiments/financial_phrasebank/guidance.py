"""FPB labeling guidance, written from the 20 seeds and the plain labeler's errors on dev half A only."""

G1 = """Annotators of this dataset (finance professionals) label from an investor's point of view and are generous with
"positive": any concrete development that benefits the company counts, even when stated in neutral factual language.
- positive: completed or agreed acquisitions/divestments that strengthen the business, signed contracts, orders, supply deals
  or secured deliveries, awards and recognition, expansion or new market entry, rising sales/profit/market share/stock price,
  favourable comparisons with a prior period or index, legal cases won or charges dismissed, the company described as leading,
  strong, well-managed or doing well, confidence of customers.
- negative: falling results, losses, layoffs, closures by the company, lawsuits or disputes against it, uncertainty clouding
  the outlook, falling stock price, missed expectations.
- neutral: purely descriptive facts with no implied benefit or harm (what a company does, where it is based, who owns what,
  product specs, plans or possibilities that are not yet decided, routine procedural announcements, third-party speculation,
  cautious remarks that are not clearly negative).
When a sentence reports a concrete business event, decide whether an investor would see it as good or bad news for the company
before falling back to neutral."""
