# eng2gloss-api

This is the repository for a JSON web api that makes use of a transformer based ML model to translate English to ASL Gloss.

To use it, first start the fastapi server (or use this project's deployment url) and make a GET request to the `/translate` endpoint with the `english_sentence`
query parameters specified. A response containing the predicted Gloss translation will be returned in a JSON format.
