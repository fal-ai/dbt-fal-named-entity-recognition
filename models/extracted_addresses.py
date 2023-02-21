def model(dbt, fal):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline
    from pathlib import Path
    from datasets import Dataset
    import pandas as pd

    dbt.config(fal_environment="named_entity_recognition", fal_machine="GPU")

    cache_dir = Path("/data/huggingface/large-ner").expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-large-NER", cache_dir=cache_dir
    )

    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device="cuda:0")

    df = dbt.ref("free_form_text")
    dataset = Dataset.from_pandas(df)

    ner_results = nlp(dataset["sentences"])
    return pd.DataFrame({"results" : ner_results})
