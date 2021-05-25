import model_ops

model, tfidf = model_ops.prepare_model()
model_ops.save(model, tfidf)
