from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask.ext.jsonpify import jsonify

app = Flask(__name__)
api = Api(app)

db_connect = create_engine('sqlite:///chinook.db')
conn = db_connect.connect() #kết nối với cơ sở dữ liệu

class Employees(Resource):
    def get(self):
        query = conn.execute("select * from employees") # Dòng này thực hiện truy vấn và trả về json
        return {'employees': [i[0] for i in query.cursor.fetchall()]} # Tìm và thêm cột đầu tiên là Employee ID

class Tracks(Resource):
    def get(self):
        query = conn.execute("select trackid, name, composer, unitprice from tracks;")
        result = {'data': [dict(zip(tuple (query.keys()) ,i)) for i in query.cursor]}
        return jsonify(result)

class Employees_Name(Resource):
    def get(self, employee_id):
        query = conn.execute("select * from employees where EmployeeId =%d "  %int(employee_id))
        result = {'data': [dict(zip(tuple (query.keys()) ,i)) for i in query.cursor]}
        return jsonify(result)
        

api.add_resource(Employees, '/employees') # Route_1
api.add_resource(Tracks, '/tracks') # Route_2
api.add_resource(Employees_Name, '/employees/<employee_id>') # Route_3

if __name__ == '__main__':
     app.run()