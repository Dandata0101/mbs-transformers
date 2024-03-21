from flask import Flask

app = Flask(__name__)

@app.before_first_request
def before_first_request():
    print("This function runs before the first request is handled.")

@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
