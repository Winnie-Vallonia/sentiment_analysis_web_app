# Importing flask and necessary libraries

# Flask: to create the web server

# request: to get data people send (like the review text)

# jsonify: to send a clean JSON response back

# load_model + joblib: to load your saved model and vectorizer


from flask import Flask, request, jsonify 
from tensorflow.keras.models import load_model
import joblib

#Initializing the Flask app
app = Flask(__name__) #Thuis creates the web server

#Loading the saved model and vectorizer
model = load_model("sentiment_model.keras")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

#Creating a flask route to handle predictions

#Define the '/analyze' route, and use the POST methode because we expect the user to send data, not just visit the page
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    #Get the text the user sends, then pull out the value under the key "text" to get the actual review, use "" if empty
        data = request.get_json()
        text = data.get("text", "")
        
        #If the user didn't end a revuiew, we return an error response
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        #Transform the raw review (string) into numbers that the model understands, using the TF-IDF vectorizer
        vectorized = vectorizer.transform([text])
        
        #Use the model to make the prediction. It returns a probability between 0 and 1
        #Closer to 1 = positive, closer to 0 = negative
        prediction = model.predict(vectorized.toarray())[0][0]
        
        #Convert predictions into words and round the confidence to 2 decimals
        sentiment = "positive" if prediction >= 0.5 else "negative"
        confidence = round(float(prediction), 2)
        
        #Send back the result in JSON format, so teh user can understand
        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence
        })
        
#Run the app for testing
if __name__ == "__main__":
    app.run(debug=True)





        

