# dbt-fal Named Entity Recognition Example

This is an example of extracting physical addresses from free-form text fields using a combination of dbt and dbt-fal. We build a dbt Python model using the dbt-fal adapter which lets us run Python models with all databases (including Bigquery and Postgres). This Python model loads a Named Entity Recognition machine learning model that extracts addresses (and some other useful metadata like persons and organizations) from free-form text.

To get access to fal Cloud, let us know!

For more info, go to the accompanying blog post: https://blog.fal.ai/building-language-model-powered-pipelines-with-dbt/
