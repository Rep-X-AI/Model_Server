from flask import Flask, jsonify,request
from teacher_entry import create_assignment
from student_submit import get_answer
from teacher_delete import delete_assignment

# Create Flask application
app = Flask(__name__)

# Define a route for the root URL
@app.route('/')
def hello():
    return "Hello, World!"

# Define a route to return JSON data
@app.route('/data')
def get_data():
    data = {'message': 'This is JSON data'}
    return jsonify(data)

# Define a route with dynamic URL parameter
@app.route('/user/<username>')
def get_user(username):
    return f"Hello, {username}!"



@app.route('/createAssignment', methods=['POST'])
def createAssignment():
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    assignment_code = data.get('assignmentCode')
    model_answer = data.get('modelAnswer')
    model_diagram = data.get('modelDiagram')


    if not assignment_code or not model_answer or not model_diagram:
        return jsonify({'error': 'Please provide all required data'}), 400

    isCreated=create_assignment(assignment_code, model_answer, model_diagram)

    if isCreated==1:
        return jsonify({'message': 'Assignment stored successfully'}), 200
    else:
        return jsonify({'message': 'Assignment already exists'}), 400



@app.route('/evaluateAssignment', methods=['POST'])
def evaluateAssignment():
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    assignment_code = data.get('assignmentCode')
    student_answer = data.get('studentAnswer')
    total_marks = data.get('totalMarks')

    if not assignment_code or not student_answer or not total_marks:
        return jsonify({'error': 'Please provide all required data'}), 400

    obtained_marks=get_answer(assignment_code, student_answer, total_marks)

    if obtained_marks!=None:

        return jsonify({'message': 'Evaluated successfully', 'obtainedMarks':obtained_marks}), 200
    else:
        return jsonify({'message': 'Invalid assignment code', 'obtainedMarks':obtained_marks}), 400



@app.route('/deleteAssignment', methods=['POST'])
def deleteAssignment():
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    assignment_code = data.get('assignmentCode')


    if not assignment_code:
        return jsonify({'error': 'Please provide all required data'}), 400

    isDeleted=delete_assignment(assignment_code)
    if isDeleted==1:
        return jsonify({'message': 'Assignment deleted successfully',}), 200
    else:
        return jsonify({'message': 'Assignment does not exists',}), 400
    


if __name__ == '__main__':
    app.run(debug=True)
    # Run the Flask ap