from flask import Flask
app= Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return "Hello, World This is a simple ping endpoint!"
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
 