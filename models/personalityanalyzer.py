import nltk
from textblob import TextBlob


def analyze_personality(text):
    """
    Analyzes the personality traits based on input text.
    This is a basic implementation using sentiment and tone analysis.
    """
    # Analyze sentiment
    blob = TextBlob(text)
    sentiment = blob.sentiment

    # Assess personality traits (simplistic approach)
    personality_traits = {
        "Openness": "You seem curious and open to new experiences." if sentiment.subjectivity > 0.5 else "You seem more grounded and factual.",
        "Agreeableness": "Your tone seems warm and friendly." if sentiment.polarity > 0.3 else "Your tone seems more neutral or reserved.",
        "Neuroticism": "You might be feeling slightly negative or anxious." if sentiment.polarity < 0 else "You seem composed and positive."
    }

    return {
        "Sentiment Analysis": f"Polarity: {sentiment.polarity:.2f}, Subjectivity: {sentiment.subjectivity:.2f}",
        "Personality Traits": personality_traits
    }


# Function to run the analysis repeatedly
def run_analysis():
    while True:
        # Take input from the user
        text_input = input("Write something about yourself (or type 'exit' to quit): ")

        # Exit the loop if the user types 'exit'
        if text_input.lower() == 'exit':
            print("Exiting analysis.")
            break

        # Analyze personality
        result = analyze_personality(text_input)

        # Display results
        print("\nAnalysis Results:")
        print(result["Sentiment Analysis"])
        for trait, description in result["Personality Traits"].items():
            print(f"{trait}: {description}")
        print("\n")

# Run the analysis
run_analysis()
