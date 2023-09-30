import pickle

def predict(data):
  model = pickle.load(open('model.pkl', 'rb'))
  return model.predict(data)
