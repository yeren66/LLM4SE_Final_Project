from langchain.tools import BaseTool
courses = [
    {"code": "CS101", "name": "计算机科学入门", "type": "必修"},
    {"code": "MA101", "name": "高等数学", "type": "必修"},
    {"code": "EN101", "name": "英语", "type": "选修"}
]

def query_courses(course_type=None):
    results = [course for course in courses if course_type is None or course['type'] == course_type]
    return results

selected_courses = []

def select_course(course_code):
    for course in courses:
        if course['code'] == course_code:
            if course in selected_courses:
                return "课程已选择。"
            selected_courses.append(course)
            return f"成功选择课程：{course['name']}"
    return "课程不存在。"

def delete_course(course_code):
    for course in selected_courses:
        if course['code'] == course_code:
            selected_courses.remove(course)
            return f"课程已删除：{course['name']}"
    return "未选择该课程。"

class QueryCourse(BaseTool):
    name = "query_courses"
    description = "use this tool when you need to query courses, optionally filtering by type"
    def _run(self, course_type=None):
        return query_courses(course_type)
    
    def _arun(self, course_type=None):
        raise NotImplementedError("This tool does not support asynchronous execution.")
    
class SelectCourse(BaseTool):
    name = "select_course"
    description = "use this tool when you need to select a course by its code"
    def _run(self, course_code):
        return select_course(course_code)
    
    def _arun(self, course_code):
        raise NotImplementedError("This tool does not support asynchronous execution.")
    
class DeleteCourse(BaseTool):
    name = "delete_course"
    description = "use this tool when you need to delete a course by its code"
    def _run(self, course_code):
        return delete_course(course_code)
    
    def _arun(self, course_code):
        raise NotImplementedError("This tool does not support asynchronous execution.")