# flow_langgraph_demo.py
from __future__ import annotations
from IPython.display import Image, display
from typing import TypedDict, List, Dict, Any, Optional, Tuple
import re, json
from dataclasses import dataclass, asdict

# ---------- Models & retrieval ----------
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ---------- LangGraph ----------
from langgraph.graph import StateGraph, END

# ---------- XML ----------
from lxml import etree

# ===============================
# 0) Activity templates (ví dụ rút gọn)
# ===============================

@dataclass
class ArgSpec:
    name: str
    type: str
    keywordArg: Optional[str] = None

@dataclass
class TplDoc:
    templateId: str
    pkg: str
    keyword: str
    text: str              # concat fields phục vụ retrieval
    requiredArgs: List[ArgSpec]

ActivityPackagesString = """[
  {
    _id: "google_drive",
    displayName: "Google Drive",
    description: "Help you integrate your work with Google Drive",
    library: "RPA.Cloud.Google",
    activityTemplates: [
      {
        templateId: "google_drive.set_up_connection",
        displayName: "Setup Drive Connection",
        description: "Set up drive connection for following task",
        iconCode: "FaEnvelope",
        type: "activity",
        keyword: "Init Drive",
        arguments: {
          Connection: {
            type: "connection.Google Drive",
            keywordArg: "token_file",
            provider: "Google Drive",
            description: "Your connection ID with Google Drive",
            value: null,
          },
        },
      },
      {
        templateId: "drive.create_folder",
        displayName: "Create folder",
        description: "Create a Google Drive folder in a given directory",
        iconCode: "FaGoogleDrive",
        type: "activity",
        keyword: "Create Drive Directory",
        arguments: {
          "Folder name": {
            type: "string",
            description: "The name of the folder",
            keywordArg: "folder",
            value: "",
          },
          "Parent Folder Path": {
            type: "string",
            description: "The path to the parent folder",
            keywordArg: "parent_folder",
            value: "",
          },
        },
        return: {
          displayName: "Folder",
          type: "dictionary",
          description:
            "The created folder. This is a dictionary, contains: id (folder id), url (folder url)",
        },
      },
      {
        templateId: "drive.dowload_files",
        displayName: "Dowload Files",
        description: "Dowload Files From Drive Folders",
        iconCode: "FaGoogleDrive",
        type: "activity",
        keyword: "Download Drive Files",
        arguments: {
          "Folder name": {
            type: "string",
            description: "The name of the folder",
            keywordArg: "source",
            value: "",
          },
          Query: {
            type: "string",
            description: "Define the file type to dowload",
            keywordArg: "query",
            value: "",
          },
        },
        return: {
          displayName: "Files",
          type: "list",
          description: "List of dowloaded files 's name",
        },
      },
      {
        templateId: "drive.upload_file",
        displayName: "Upload file",
        description: "Upload a file from robot's file system to Google Drive",
        iconCode: "FaGoogleDrive",
        type: "activity",
        keyword: "Upload Drive File",
        arguments: {
          "File name": {
            type: "string",
            keywordArg: "filename",
            value: "",
          },
          "Folder Path": {
            type: "string",
            keywordArg: "folder",
            value: "",
          },
          Overwrite: {
            type: "boolean",
            keywordArg: "overwrite",
            value: false,
          },
          "Make Folder": {
            type: "boolean",
            keywordArg: "make_dir",
            value: false,
          },
        },
        return: {
          displayName: "File id",
          type: "string",
          description: "The uploaded file id",
        },
      },
      {
        templateId: "drive.get_file_list_in_folder",
        displayName: "Get file list in folder",
        description: "Get a list of files in a given folder in Google Drive",
        iconCode: "FaGoogleDrive",
        type: "activity",
        keyword: "Search Drive Files",
        arguments: {
          "Folder Path": {
            type: "string",
            description: "The path to the folder",
            keywordArg: "source",
            value: "",
          },
          Query: {
            type: "string",
            description: "Enter your query condition",
            keywordArg: "query",
            value: "",
          },
        },
        return: {
          displayName: "File List",
          type: "list",
          description:
            "A list of files. Each file is a dictionary, contains: id (file id), url (file url), name (file name), is_folder, mimeType (file mime type), size (file size), modifiedTime (file modified time)",
        },
      },
      {
        templateId: "drive.get_file_folder",
        displayName: "Get a file/folder",
        description: "Get a file/folder in Google Drive",
        iconCode: "FaGoogleDrive",
        type: "activity",
        keyword: "Get Drive File By Id",
        arguments: {
          ID: {
            type: "string",
            description: "The ID of folder or file",
            keywordArg: "file_id",
            value: "",
          },
        },
        return: {
          displayName: "File/Folder",
          type: "dictionary",
          description:
            "The file/folder. This is a dictionary, contains: id (file/folder id), url (file/folder url), name (file/folder name), is_folder, mimeType (file/folder mime type), size (file/folder size), modifiedTime (file/folder modified time)",
        },
      },
      {
        templateId: "drive.delete_file_folder",
        displayName: "Delete file/folder",
        description: "Delete a file/folder in Google Drive",
        iconCode: "FaGoogleDrive",
        type: "activity",
        keyword: "Delete Drive File",
        arguments: {
          ID: {
            type: "string",
            description: "The ID of folder or file",
            keywordArg: "file_id",
            value: "",
          },
        },
        return: {
          displayName: "Number of deleted",
          type: "number",
          description: "The number of deleted files/folders",
        },
      },
      {
        templateId: "drive.move_file_folder",
        displayName: "Move file/folder",
        description: "Move a file/folder to another folder in Google Drive",
        iconCode: "FaGoogleDrive",
        type: "activity",
        keyword: "Move Drive File",
        arguments: {
          "Source ID": {
            type: "string",
            description: "The ID of source folder or file",
            keywordArg: "file_id",
            value: "",
          },
          "Destination Folder Path": {
            type: "string",
            description: "The path to destination folder",
            keywordArg: "target",
            value: "",
          },
        },
        return: {
          displayName: "List of files/folders id",
          type: "list",
          description: "A list of files/folders id",
        },
      },
      {
        templateId: "drive.share_file_folder",
        displayName: "Share a file/folder",
        description: "Share a file/folder in Google Drive",
        iconCode: "FaGoogleDrive",
        type: "activity",
        keyword: "Add Drive Share",
        arguments: {
          "Share Type": {
            type: "enum.shareType",
            description: "Share with list emails or all people",
            keywordArg: "share_type",
            value: "user",
          },
          "Share with Email": {
            type: "email",
            description: "Email address to share with",
            keywordArg: "email",
            value: "",
          },
          Permission: {
            type: "enum.permission",
            description: "The role including reader, commenter, writer",
            keywordArg: "role",
            value: "reader",
          },
          ID: {
            type: "string",
            description: "The ID of the file or folder",
            keywordArg: "file_id",
            value: "",
          },
        },
        return: {
          displayName: "Share response",
          type: "dictionary",
          description:
            "The share response. This is a dictionary, contains: file_id, permission_id",
        },
      },
    ],
  },
  {
    _id: "gmail",
    displayName: "Gmail",
    description: "Help you integrate your work with Gmail",
    library: "RPA.Cloud.Google",
    activityTemplates: [
      {
        templateId: "gmail.set_up_connection",
        displayName: "Setup Gmail Connection",
        description: "Set up Gmail connection for following task",
        iconCode: "FaEnvelope",
        type: "activity",
        keyword: "Init Gmail",
        arguments: {
          Connection: {
            type: "connection.Gmail",
            keywordArg: "token_file",
            provider: "Gmail",
            description: "Your connection ID with Gmail",
            value: null,
          },
        },
      },
      {
        templateId: "gmail.send_email",
        displayName: "Send email",
        description: "Send an email to other people using Gmail",
        iconCode: "FaEnvelope",
        type: "activity",
        keyword: "Send Message",
        arguments: {
          From: {
            type: "string",
            description: "Your source email",
            keywordArg: "sender",
            value: "me",
          },
          To: {
            type: "email",
            description: "Email you want to send email to",
            keywordArg: "to",
            value: "",
          },
          Subject: {
            type: "string",
            description: "The subject of email",
            keywordArg: "subject",
            value: "",
          },
          Body: {
            type: "string",
            description: "The body of email",
            keywordArg: "message_text",
            value: "",
          },
        },
        return: {
          displayName: "Sent message",
          type: "dictionary",
          description:
            "The sent message. This is a dictionary, contains: id (message id), threadId (message thread id)",
        },
      },
      {
        templateId: "gmail.list_emails",
        displayName: "Get list emails",
        description: "List emails in a given folder in Gmail",
        iconCode: "FaEnvelope",
        type: "activity",
        keyword: "List Messages",
        arguments: {
          "Email Folder Path": {
            type: "string",
            description: "The source email folder path",
            keywordArg: "label_ids",
            value: [],
          },
          "User ID": {
            type: "string",
            description: "The ID of user",
            keywordArg: "user_id",
            value: "me",
          },
          Query: {
            type: "string",
            description: "The query condition",
            keywordArg: "query",
            value: "",
          },
          "Max number emails": {
            type: "number",
            description: "Filter by the limit number of emails",
            keywordArg: "max_results",
            value: 100,
          },
        },
        return: {
          displayName: "Emails",
          type: "list",
          description:
            "A list of emails. Each email is a dictionary, contains: id (email id), from (email from), to (email to), cc (email cc), bcc (email bcc), subject (email subject), body (email body), attachments (email attachments)",
        },
      },
    ],
  },
  {
    _id: "google_sheets",
    displayName: "Google Sheet",
    description: "Help you integrate your work with Google Sheets",
    library: "RPA.Cloud.Google",
    activityTemplates: [
      {
        templateId: "google_sheets.set_up_connection",
        displayName: "Setup Google Sheet Connection",
        description: "Set up Google Sheet connection for following task",
        iconCode: "FaEnvelope",
        type: "activity",
        keyword: "Init Sheets",
        arguments: {
          Connection: {
            type: "connection.Google Sheets",
            keywordArg: "token_file_path",
            provider: "Google Sheets",
            description: "Your connection ID with Google Sheet",
            value: null,
          },
        },
      },
      {
        templateId: "sheet.create_spreadsheet",
        displayName: "Create SpreadSheet",
        description: "Create SpreadSheet in Google Sheet",
        iconCode: "FaFileSpreadsheet",
        type: "activity",
        keyword: "Create Spreadsheet",
        arguments: {
          "SpreadSheet Name": {
            type: "string",
            description: "The spread sheet name",
            keywordArg: "title",
            value: "",
          },
        },
        return: {
          displayName: "SpreadSheet ID",
          type: "string",
          description: "The created spreadsheet id",
        },
      },
      {
        templateId: "sheet.get_spreadsheet_by_id",
        displayName: "Get SpreadSheet By Id",
        description: "Get SpreadSheet By Id in Google Sheet",
        iconCode: "FaFileSpreadsheet",
        type: "activity",
        keyword: "Get Spreadsheet Basic Information",
        arguments: {
          "SpreadSheet ID": {
            type: "string",
            description: "The ID of spread sheet",
            keywordArg: "spreadsheet_id",
            value: "",
          },
        },
        return: {
          displayName: "SpreadSheet",
          type: "dictionary",
          description:
            "The spreadsheet. This is a dictionary, contains: id (spreadsheet id), url (spreadsheet url), name (spreadsheet name), sheets (spreadsheet sheets)",
        },
      },
      {
        templateId: "sheet.add_sheet",
        displayName: "Add sheet",
        description: "Add sheet to a given SpreadSheet in Google Sheet",
        iconCode: "FaFileSpreadsheet",
        type: "activity",
        keyword: "Create Sheet",
        arguments: {
          "SpreadSheet ID": {
            type: "string",
            description: "The ID of spread sheet",
            keywordArg: "spreadsheet_id",
            value: "",
          },
          "Sheet Name": {
            type: "string",
            description: "The name of the sheet",
            keywordArg: "sheet_name",
            value: "",
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Operation result as an dictionary",
        },
      },
      {
        templateId: "sheet.delete_sheet",
        displayName: "Delete sheet",
        description: "Delete sheet from a given SpreadSheet in Google Sheet",
        iconCode: "FaFileSpreadsheet",
        type: "activity",
        keyword: "Delete Sheet",
        arguments: {
          "SpreadSheet ID": {
            type: "string",
            description: "The ID of spread sheet",
            keywordArg: "spreadsheet_id",
            value: "",
          },
          "Sheet Name": {
            type: "string",
            description: "The name of the sheet",
            keywordArg: "sheet_name",
            value: "",
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Operation result as an dictionary",
        },
      },
      {
        templateId: "sheet.rename_sheet",
        displayName: "Rename sheet",
        description: "Rename sheet of a given SpreadSheet in Google Sheet",
        iconCode: "FaFileSpreadsheet",
        type: "activity",
        keyword: "Rename Sheet",
        arguments: {
          "SpreadSheet ID": {
            type: "string",
            description: "The ID of spread sheet",
            keywordArg: "spreadsheet_id",
            value: "",
          },
          "Old Sheet Name": {
            type: "string",
            description: "The old name of sheet",
            keywordArg: "sheet_name",
            value: "",
          },
          "New Sheet Name": {
            type: "string",
            description: "The new name of sheet",
            keywordArg: "new_sheet_name",
            value: "",
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Operation result as an dictionary",
        },
      },
      {
        templateId: "sheet.write_data_to_sheet",
        displayName: "Write Data To Sheet",
        description:
          "Write Data To Sheet in a given SpreadSheet in Google Sheet",
        iconCode: "FaFileSpreadsheet",
        type: "activity",
        keyword: "Update Sheet Values",
        arguments: {
          "SpreadSheet ID": {
            type: "string",
            description: "The ID of spread sheet",
            keywordArg: "spreadsheet_id",
            value: "",
          },
          "Sheet Range": {
            type: "string",
            description: "The range of the sheet",
            keywordArg: "sheet_range",
            value: "",
          },
          Content: {
            type: "string",
            description: "The data written to the sheet",
            keywordArg: "values",
            value: [],
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Operation result",
        },
      },
      {
        templateId: "sheet.read_data_from_sheet",
        displayName: "Read Data From Sheet",
        description:
          "Read Data From Sheet in a given SpreadSheet in Google Sheet",
        iconCode: "FaFileSpreadsheet",
        type: "activity",
        keyword: "Get Sheet Values",
        arguments: {
          "SpreadSheet ID": {
            type: "string",
            description: "The ID of spread sheet",
            keywordArg: "spreadsheet_id",
            value: "",
          },
          "Sheet Range": {
            type: "string",
            description: "The range of the sheet",
            keywordArg: "sheet_range",
            value: "",
          },
        },
        return: {
          displayName: "Sheet Values",
          type: "list",
          description: "A list of values. Each value is a list of cells value",
        },
      },
      {
        templateId: "sheet.clear_data_from_sheet",
        displayName: "Clear Data From Sheet",
        description:
          "Clear Data From Sheet in a given SpreadSheet in Google Sheet",
        iconCode: "FaFileSpreadsheet",
        type: "activity",
        keyword: "Clear Sheet Values",
        arguments: {
          "SpreadSheet ID": {
            type: "string",
            description: "The ID of spread sheet",
            keywordArg: "spreadsheet_id",
            value: "",
          },
          "Sheet Range": {
            type: "string",
            description: "The range of the sheet",
            keywordArg: "sheet_range",
            value: "",
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Operation result",
        },
      },
    ],
  },
  {
    _id: "google_classroom",
    displayName: "Google Classroom",
    description: "Help you integrate your work with Google Classroom",
    library: "EduRPA.Google",
    activityTemplates: [
      {
        templateId: "google_classroom.set_up_connection",
        displayName: "Setup Google Classroom Connection",
        description: "Set up Google Classroom connection for following task",
        iconCode: "FaEnvelope",
        type: "activity",
        keyword: "Set Up Classroom Connection",
        arguments: {
          Librabry: {
            type: "string",
            value: "EduRPA.Google",
            description: "Librabry for setup OAuth token",
            hidden: true,
          },
          Connection: {
            type: "connection.Google Classroom",
            description: "Your connection ID with Google Classroom",
            keywordArg: "token_file_path",
            provider: "Google Classroom",
            value: null,
          },
        },
      },
      {
        templateId: "create_course",
        displayName: "Create Course",
        description: "Create new course for teacher",
        type: "activity",
        keyword: "Create Course",
        arguments: {
          "Course Name": {
            type: "string",
            keywordArg: "name",
            description: "Name of the created course",
            value: "",
          },
          "Teacher Email": {
            type: "string",
            keywordArg: "ownerId",
            description: "Email of teacher you would to invite",
            value: "",
          },
        },
        return: {
          displayName: "Course ID",
          type: "string",
          description: "The ID of the course",
        },
      },
      {
        templateId: "list_classrooms",
        displayName: "List Classrooms",
        description: "List Classrooms",
        type: "activity",
        keyword: "List Classrooms",
        arguments: {},
        return: {
          displayName: "List of Classrooms",
          type: "list",
          description: "List of dictionary of course object with {name, id}",
        },
      },
      {
        templateId: "delete_course_by_id",
        displayName: "Delete Classroom",
        description: "Delete Classroom",
        type: "activity",
        keyword: "Delete Classroom",
        arguments: {
          "Course ID": {
            type: "string",
            keywordArg: "courseId",
            description: "ID of the course",
            value: "",
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Operation result",
        },
      },
      {
        templateId: "get_course_id_by_course_name",
        displayName: "Get Course ID By Course Name",
        description: "Get ID of the course by course name",
        type: "activity",
        keyword: "Get Course ID By Course Name",
        arguments: {
          "Course Name": {
            type: "string",
            keywordArg: "course_name",
            description: "Name of the course",
            value: "",
          },
        },
        return: {
          displayName: "Course ID",
          type: "string",
          description: "The ID of the course",
        },
      },
      {
        templateId: "invite_student_course",
        displayName: "Invite Students To Classroom",
        description: "Invite Students To Classroom",
        type: "activity",
        keyword: "Invite Students To Classroom",
        arguments: {
          "Course ID": {
            type: "string",
            keywordArg: "courseId",
            description: "ID of the course",
            value: "",
          },
          "List of student emails": {
            type: "list",
            keywordArg: "studentEmails",
            description: "List of student emails",
            value: "",
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Operation result",
        },
      },
      {
        templateId: "create_assignment",
        displayName: "Create Assignment",
        description: "Create Assignment in a course of Google Classroom",
        type: "activity",
        keyword: "Create Assignment",
        arguments: {
          "Course ID": {
            type: "string",
            keywordArg: "courseId",
            description: "ID of the course",
            value: "",
          },
          "Assignment Title": {
            type: "string",
            keywordArg: "title",
            description: "Title of the assignment",
            value: "",
          },
          "Assignment Description": {
            type: "string",
            keywordArg: "description",
            description: "Description of the assignment",
            value: "",
          },
          "Assignment URL": {
            type: "list",
            keywordArg: "linkMaterials",
            description: "URL of the assignment",
            value: "",
          },
          "Due Date": {
            type: "string",
            keywordArg: "dueDate",
            description: "Due date of the assignment",
            value: "",
          },
          "Due Time": {
            type: "string",
            keywordArg: "dueTime",
            description: "Due time of the assignment",
            value: "",
          },
        },
        return: {
          displayName: "ID of Course Assignment",
          type: "string",
          description: "The ID of Course Assignment",
        },
      },
      {
        templateId: "create_quiz_classroom",
        displayName: "Create Quiz",
        description: "Create Quiz in a course of Google Classroom",
        type: "activity",
        keyword: "Create Quiz",
        arguments: {
          "Course ID": {
            type: "string",
            keywordArg: "courseId",
            description: "ID of the course",
            value: "",
          },
          "Quiz Title": {
            type: "string",
            keywordArg: "title",
            description: "Title of the quiz",
            value: "",
          },
          "Quiz Description": {
            type: "string",
            keywordArg: "description",
            description: "Description of the quiz",
            value: "",
          },
          "Quiz URL": {
            type: "string",
            keywordArg: "quizUrl",
            description: "URL of the quiz",
            value: "",
          },
          "Max Points": {
            type: "number",
            keywordArg: "maxPoints",
            description: "Maximum points of the quiz",
          },
          "Due Date (Optional)": {
            type: "string",
            keywordArg: "dueDate",
            description: "Due date of the assignment",
            value: "",
          },
          "Due Time (Optional)": {
            type: "string",
            keywordArg: "dueTime",
            description: "Due time of the assignment",
            value: "",
          },
        },
        return: {
          displayName: "ID of Course Quiz",
          type: "string",
          description: "The ID of Course Quiz",
        },
      },
      {
        templateId: "list_course_work",
        displayName: "List Coursework",
        description: "List Coursework",
        type: "activity",
        keyword: "List Coursework",
        arguments: {
          "Course ID": {
            type: "string",
            keywordArg: "courseId",
            description: "ID of the course",
            value: "",
          },
        },
        return: {
          displayName: "List of Coursework In Course",
          type: "list",
          description: "List of Coursework In Course",
        },
      },
      {
        templateId: "get_coursework_id_by_title",
        displayName: "Get Coursework ID By Title",
        description: "Get Coursework ID By Title",
        type: "activity",
        keyword: "Get Coursework ID By Title",
        arguments: {
          "Course ID": {
            type: "string",
            keywordArg: "courseId",
            description: "ID of the course",
            value: "",
          },
          "Course Title": {
            type: "string",
            keywordArg: "title",
            description: "Title of the course",
            value: "",
          },
        },
        return: {
          displayName: "Coursework ID of the course",
          type: "string",
          description: "Coursework ID of the course",
        },
      },
      {
        templateId: "delete_coursework",
        displayName: "Delete Coursework",
        description: "Delete Coursework",
        type: "activity",
        keyword: "Delete Coursework",
        arguments: {
          "Course ID": {
            type: "string",
            keywordArg: "courseId",
            description: "ID of the course",
            value: "",
          },
          "Coursework ID": {
            type: "string",
            keywordArg: "courseworkId",
            description: "ID of the course work",
            value: "",
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Operation result",
        },
      },
      {
        templateId: "list_student_submissions",
        displayName: "List Student Submissions",
        description: "List Student Submissions",
        type: "activity",
        keyword: "List Student Submissions",
        arguments: {
          "Course ID": {
            type: "string",
            keywordArg: "courseId",
            description: "ID of the course",
            value: "",
          },
          "Coursework ID": {
            type: "string",
            keywordArg: "courseworkId",
            description: "ID of the coursework",
            value: "",
          },
        },
        return: {
          displayName: "Student submissions",
          type: "list",
          description: "List of student submissions of the coursework",
        },
      },
      {
        templateId: "get_submission_id_by_email",
        displayName: "Get Submission ID By Email",
        description: "Get Submission ID By Email",
        type: "activity",
        keyword: "Get Submission ID By Email",
        arguments: {
          "Course ID": {
            type: "string",
            keywordArg: "courseId",
            description: "ID of the course",
            value: "",
          },
          "Coursework ID": {
            type: "string",
            keywordArg: "courseworkId",
            description: "ID of the coursework",
            value: "",
          },
          "Student Email": {
            type: "string",
            keywordArg: "studentEmail",
            description: "Email of the student",
            value: "",
          },
        },
        return: {
          displayName: "ID of the submission",
          type: "string",
          description: "ID of the submission",
        },
      },
    ],
  },
  {
    _id: "google_form",
    displayName: "Google Form",
    description: "Help you integrate your work with Google Form",
    library: "EduRPA.Google",
    activityTemplates: [
      {
        templateId: "google_form.set_up_connection",
        displayName: "Setup Google Form Connection",
        description: "Set up Google Form connection for following task",
        iconCode: "FaEnvelope",
        type: "activity",
        keyword: "Set Up Form Connection",
        arguments: {
          Librabry: {
            type: "string",
            value: "EduRPA.Google",
            description: "Librabry for setup OAuth token",
            hidden: true,
          },
          Connection: {
            type: "connection.Google Form",
            keywordArg: "token_file_path",
            description: "Your connection ID with Google Form",
            provider: "Google Forms",
            value: null,
          },
        },
      },
      {
        templateId: "create_quiz_form",
        displayName: "Create Quiz Form",
        description: "Create quiz in google form",
        type: "activity",
        keyword: "Create Form",
        arguments: {
          "Form Name": {
            type: "string",
            keywordArg: "title",
            description: "Name of Google Form",
            value: "",
          },
        },
        return: {
          displayName: "ID of created quiz form",
          type: "string",
          description: "The ID of created quiz form",
        },
      },
      {
        templateId: "get_doc_id",
        displayName: "Get Google Doc ID From URL",
        description: "Get Google Doc ID from URL",
        type: "activity",
        keyword: "Get Google Doc ID",
        arguments: {
          URL: {
            type: "string",
            keywordArg: "url",
            description: "URL of Google Doc",
            value: "",
          },
        },
        return: {
          displayName: "ID of Google Doc",
          type: "string",
          description: "The ID of Google Doc",
        },
      },
      {
        templateId: "transfer_quiz",
        displayName: "Transfer Google Doc To Google",
        description: "Transfer quiz from google doc to google form",
        type: "activity",
        keyword: "Add Questions And Answers From Google Doc To Form",
        arguments: {
          DocID: {
            type: "string",
            keywordArg: "doc_id",
            description: "ID of Google Doc",
            value: "",
          },
          FormID: {
            type: "string",
            keywordArg: "form_id",
            description: "ID of Google Form",
            value: "",
          },
        },
        return: {
          displayName: "The link of Google Form",
          type: "string",
          description: "The link of Google Form",
        },
      },
    ],
  },
  {
    _id: "control",
    displayName: "Control",
    description: "Help you control the execution flow of your robot",
    activityTemplates: [
      {
        templateId: "if",
        displayName: "If/Else",
        description:
          "If a condition is met, then execute a set of activities, otherwise execute another set of activities",
        iconCode: "AiOutlineBranches",
        type: "gateway",
        arguments: {
          Condition: {
            type: "list.condition",
            description: "List of condition",
            value: "",
          },
        },
        return: null,
      },
      {
        templateId: "for_each",
        displayName: "For each",
        description: "Execute a set of activities for each item in a list",
        iconCode: "ImLoop2",
        type: "subprocess",
        arguments: {
          LoopType: {
            type: "string",
            value: "for_each",
            description: "Type to parse loop",
            hidden: true,
          },
          Item: {
            type: "string",
            description: "Iterate Variable",
            value: "",
          },
          List: {
            type: "list",
            description: "Iterate Struture",
            value: "",
          },
        },
      },
      {
        templateId: "for_range",
        displayName: "For Value In Range",
        description: "Execute a set of activities for each item in range",
        iconCode: "ImLoop2",
        type: "subprocess",
        arguments: {
          LoopType: {
            type: "string",
            value: "for_range",
            description: "Type to parse loop",
            hidden: true,
          },
          Item: {
            type: "string",
            description: "Iterate Variable",
            value: "",
          },
          Start: {
            type: "number",
            description: "start value",
            value: "",
          },
          End: {
            type: "number",
            description: "start value",
            value: "",
          },
        },
      },
    ],
  },
  {
    _id: "data_manipulation",
    displayName: "Data manipulation",
    description: "Help you manipulate data in your robot",
    library: "Collections",
    activityTemplates: [
      {
        templateId: "set_variable",
        displayName: "Set variable",
        description: "Set the value of a variable",
        iconCode: "FaEquals",
        type: "activity",
        keyword: "Set Variable",
        arguments: {
          Variable: {
            type: "variable",
            description: "The variable to set the value to",
            keywordArg: "variable",
            value: "",
          },
          Value: {
            type: "any",
            description: "The value to set to the variable",
            keywordArg: "value",
            value: "",
          },
        },
        return: null,
      },
      {
        templateId: "add_to_list",
        displayName: "Add to list",
        description: "Add an item to a list",
        iconCode: "FaListUl",
        type: "activity",
        keyword: "Append To List",
        arguments: {
          List: {
            type: "list",
            description: "The list",
            keywordArg: 'list_',
            overrideType: RFVarType["any"],
            value: [],
          },
          Item: {
            type: "any",
            description: "The item to add to the list",
            overrideType: RFVarType["any"],
            value: "",
          },
        },
        return: null,
      },
      {
        templateId: "remove_from_list",
        displayName: "Remove from list",
        description: "Remove an item from a list",
        iconCode: "FaListUl",
        type: "activity",
        keyword: "Remove From List",
        arguments: {
          List: {
            type: "list",
            description: "The list",
            keywordArg: "list",
            value: [],
          },
          Item: {
            type: "any",
            description: "The item to remove from the list",
            keywordArg: "item",
            value: "",
          },
        },
        return: null,
      },
      {
        templateId: "clear_list",
        displayName: "Clear list",
        description: "Clear all items in a list",
        iconCode: "FaListUl",
        type: "activity",
        keyword: "Clear List",
        arguments: {
          List: {
            type: "list",
            description: "The list",
            keywordArg: "list",
            value: [],
          },
        },
        return: null,
      },
    ],
  },
  {
    _id: "browser_automation",
    displayName: "Browser automation",
    description:
      "Help you automate tasks that need to be done in a web browser (like Chrome)",
    library: "RPA.Browser.Playwright",
    activityTemplates: [
      {
        templateId: "go_to_url",
        displayName: "Go to URL",
        description: "Go to a given URL in the current browser tab",
        iconCode: "GoBrowser",
        type: "activity",
        keyword: "Go To",
        arguments: {
          URL: {
            type: "string",
            description: "The URL link",
            keywordArg: "url",
            value: "",
          },
        },
        return: null,
      },
      {
        templateId: "click",
        displayName: "Click",
        description: "Click on a given element in the current browser tab",
        iconCode: "FaMousePointer",
        type: "activity",
        keyword: "Click",
        arguments: {
          Element: {
            type: "string",
            description: "The element HTML DOM of the website",
            keywordArg: "selector",
            value: "",
          },
        },
        return: null,
      },
      {
        templateId: "type",
        displayName: "Type Into",
        description:
          "Type a given text into a given element in the current browser tab",
        iconCode: "FaKeyboard",
        type: "activity",
        keyword: "Fill Text",
        arguments: {
          Element: {
            type: "string",
            description: "The HTML DOM element of the website",
            keywordArg: "selector",
            value: "",
          },
          Text: {
            type: "string",
            description: "The text to type to the website",
            keywordArg: "txt",
            value: "",
          },
        },
        return: null,
      },
      {
        templateId: "get_text",
        displayName: "Get text",
        description:
          "Get the text of a given element in the current browser tab",
        iconCode: "FaFont",
        type: "activity",
        keyword: "Get Text",
        arguments: {
          Element: {
            type: "string",
            description: "The HTML DOM element of the website",
            keywordArg: "selector",
            value: "",
          },
        },
        return: {
          displayName: "Text",
          type: "string",
          description: "The text of the element",
        },
      },
    ],
  },
  {
    _id: "document_automation",
    displayName: "Document automation",
    description:
      "Help you automate tasks related to documents (traditional paper documents or digital documents like PDFs) with the help of AI",
    library: "EduRPA.Document",
    activityTemplates: [
      {
        templateId: "extract_data_from_document",
        displayName: "Extract data from document",
        description: "Extract data from a document using Document template",
        iconCode: "FaFileAlt",
        type: "activity",
        keyword: "Extract Data From Document",
        arguments: {
          Document: {
            type: "string",
            description: "The document file name to extract data from",
            keywordArg: "file_name",
            value: "",
          },
          "Document template": {
            type: "DocumentTemplate",
            description: "The document template",
            keywordArg: "template",
            value: "",
          },
        },
        return: {
          displayName: "Data",

          type: "dictionary",
          description: "The extracted data from the document",
        },
      },
      {
        templateId: "generate_grade_report",
        displayName: "Generate grade report",
        description: "Generate a grade report from a list of extracted data",
        iconCode: "FaFileAlt",
        type: "activity",
        keyword: "Create Grade Report File",
        arguments: {
          "Actual answers": {
            type: "list",
            description: "The list of extracted data",
            keywordArg: "actual_answers",
            value: [],
          },
          "Correct answer": {
            type: "dictionary",
            description: "The correct answer",
            keywordArg: "correct_answer",
            value: {},
          },
          Names: {
            type: "list",
            description: "The list of student names",
            keywordArg: "file_names",
            value: [],
          },
        },
        return: {
          displayName: "Grade report file name",

          type: "string",
          description: "The generated grade report file name",
        },
      },
    ],
  },
  {
    _id: "file_storage",
    displayName: "File storage",
    description:
      "Help you store and retrieve files in the platform's file storage",
    library: "EduRPA.Storage",
    activityTemplates: [
      {
        templateId: "upload_file",
        displayName: "Upload file",
        description: "Upload a file to the platform's file storage",
        iconCode: "FaFileUpload",
        type: "activity",
        keyword: "Upload File",
        arguments: {
          File: {
            type: "string",
            description: "The file to upload",
            keywordArg: "file",
            value: "",
          },
          "File name": {
            type: "string",
            description: "The name of the file",
            keywordArg: "file_name",
            value: "",
          },
          "Folder path": {
            type: "string",
            description: "The path of the folder to store the file",
            keywordArg: "folder_path",
            value: "",
          },
        },
        return: {
          displayName: "File path",

          type: "string",
          description: "The uploaded file path",
        },
      },
      {
        templateId: "download_file",
        displayName: "Download file",
        description: "Download a file from the platform's file storage",
        iconCode: "FaFileDownload",
        type: "activity",
        keyword: "Download File",
        arguments: {
          "File path": {
            type: "string",
            description: "The path of the file to download",
            keywordArg: "file_path",
            value: "",
          },
          "File name": {
            type: "string",
            description: "The name of the file to download",
            keywordArg: "file_name",
            value: "",
          },
        },
        return: {
          displayName: "File name",
          type: "string",
          description: "The downloaded file name",
        },
      },
    ],
  },
  {
    _id: "rpa-sap-mock",
    displayName: "SAP MOCK",
    description: "Help you to handle sap activities",
    library: "RPA.MOCK_SAP",
    activityTemplates: [
      {
        templateId: "connect_to_sap_system",
        displayName: "Connect to SAP System",
        description:
          "Connect to the SAP system using a base URL and token file",
        iconCode: "FaLink",
        type: "activity",
        keyword: "Connect To SAP System",
        arguments: {
          "Base URL": {
            type: "string",
            description: "The base URL of the SAP system",
            keywordArg: "base_url",
            value: "",
          },
          "Token File Path": {
            type: "connection.SAP Mock",
            description: "The path to the file containing the SAP access token",
            keywordArg: "token_file_path",
            value: "",
          },
          "Verify SSL": {
            type: "boolean",
            description: "Whether to verify SSL certificates",
            keywordArg: "verify_ssl",
            value: false,
          },
        },
        return: {
          displayName: "Connection Status",
          type: "void",
          description: "Indicates successful connection to the SAP system",
        },
      },
      {
        templateId: "get_business_partner",
        displayName: "Get Business Partner",
        description: "Retrieve a business partner by ID from the SAP system",
        iconCode: "FaUser",
        type: "activity",
        keyword: "Get Business Partner By ID",
        arguments: {
          "Partner ID": {
            type: "string",
            description: "The ID of the business partner to retrieve",
            keywordArg: "partner_id",
            value: "",
          },
        },
        return: {
          displayName: "Business Partner Data",
          type: "object",
          description:
            "The business partner data retrieved from the SAP system",
        },
      },
      {
        templateId: "create_business_partner_address",
        displayName: "Create Business Partner Address",
        description:
          "Create a new address for a business partner in the SAP system",
        iconCode: "FaAddressCard",
        type: "activity",
        keyword: "Create Business Partner Address",
        arguments: {
          "Partner ID": {
            type: "string",
            description: "The ID of the business partner",
            keywordArg: "partner_id",
            value: "",
          },
          "JSON Data": {
            type: "string",
            description: "The address data in JSON format",
            keywordArg: "json_data",
            value: "",
          },
        },
        return: {
          displayName: "Created Address Data",
          type: "object",
          description: "The created address data returned from the SAP system",
        },
      },
      {
        templateId: "update_business_partner_address",
        displayName: "Update Business Partner Address",
        description:
          "Update an existing address for a business partner in the SAP system",
        iconCode: "FaEdit",
        type: "activity",
        keyword: "Update Business Partner Address",
        arguments: {
          "Partner ID": {
            type: "string",
            description: "The ID of the business partner",
            keywordArg: "partner_id",
            value: "",
          },
          "Address ID": {
            type: "string",
            description: "The ID of the address to update",
            keywordArg: "address_id",
            value: "",
          },
          "JSON Data": {
            type: "string",
            description: "The updated address data in JSON format",
            keywordArg: "json_data",
            value: "",
          },
        },
        return: {
          displayName: "Updated Address Data",
          type: "object",
          description: "The updated address data returned from the SAP system",
        },
      },
      {
        templateId: "delete_business_partner_address",
        displayName: "Delete Business Partner Address",
        description:
          "Delete an address for a business partner in the SAP system",
        iconCode: "FaTrash",
        type: "activity",
        keyword: "Delete Business Partner Address",
        arguments: {
          "Partner ID": {
            type: "string",
            description: "The ID of the business partner",
            keywordArg: "partner_id",
            value: "",
          },
          "Address ID": {
            type: "string",
            description: "The ID of the address to delete",
            keywordArg: "address_id",
            value: "",
          },
        },
        return: {
          displayName: "Deletion Status",
          type: "string",
          description: "The response text indicating the deletion status",
        },
      },
    ],
  },
];
"""

ActivityPackages = json.loads(ActivityPackagesString)

def build_tpl_docs() -> List[TplDoc]:
    docs: List[TplDoc] = []
    for pkg in ActivityPackages:
        for t in pkg["activityTemplates"]:
            args = t.get("arguments", {})
            required = [ArgSpec(name=k, type=v["type"], keywordArg=v.get("keywordArg")) for k, v in args.items()]
            text = " ".join([
                pkg["displayName"], t["displayName"], t["description"], t["keyword"],
                *[f"{k} {v['type']} {v.get('keywordArg','')}" for k, v in args.items()]
            ]).lower()
            docs.append(TplDoc(
                templateId=t["templateId"],
                pkg=pkg["_id"],
                keyword=t["keyword"],
                text=text,
                requiredArgs=required
            ))
    return docs

TPL_DOCS = build_tpl_docs()
TPL_INDEX = { d.templateId: d for d in TPL_DOCS }

# ===============================
# 1) State cho LangGraph
# ===============================

class FlowState(TypedDict, total=False):
    user_input: str
    normalized: str
    candidates: List[Tuple[TplDoc, float]]     # (template, score)
    chosen: List[TplDoc]                       # chọn 1-3 templates
    args_map: Dict[str, Dict[str, Any]]        # templateId -> args
    ir: Dict[str, Any]
    bpmn_xml: str
    errors: List[str]

# ===============================
# 2) Chuẩn hoá & synonyms
# ===============================

SYN = [
    ("thư", "email"),
    ("gmail", "email"),
    ("mail", "email"),
    ("ổ đĩa", "drive"),
    ("google drive", "drive"),
    ("thư mục", "folder"),
    ("tải lên", "upload"),
    ("tải xuống", "download"),
    ("ghi đè", "overwrite"),
]

def normalize_text(s: str) -> str:
    x = s.lower()
    x = re.sub(r"\s+", " ", x).strip()
    for a, b in SYN:
        x = x.replace(a, b)
    return x

# ===============================
# 3) Retrieval = BERT embedding + BM25
# ===============================

EMBED = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def embed_batch(texts: List[str]) -> List[List[float]]:
    # normalize_embeddings=True trong model này -> cosine là dot
    v = EMBED.encode(texts, normalize_embeddings=True)
    return v.tolist()

def retrieve_candidates(query: str, docs: List[TplDoc], k: int = 5) -> List[Tuple[TplDoc, float]]:
    # BM25
    corpus_tokens = [d.text.split() for d in docs]
    bm25 = BM25Okapi(corpus_tokens)
    bm_scores = bm25.get_scores(query.split())  # array

    # Embedding
    doc_emb = embed_batch([d.text for d in docs])
    qv = embed_batch([query])[0]
    def cos(a, b): return sum(aa*bb for aa, bb in zip(a, b))
    cos_scores = [cos(qv, dv) for dv in doc_emb]

    # Late fusion
    alpha = 0.7
    fused = [alpha*cos_scores[i] + (1-alpha)*bm_scores[i] for i in range(len(docs))]
    idx = sorted(range(len(docs)), key=lambda i: fused[i], reverse=True)[:k]
    return [(docs[i], float(fused[i])) for i in idx]

# ===============================
# 4) Slot filling (NER + regex/rule)
# ===============================

NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"  # đa ngữ; nếu gặp rắc rối env, dùng "dslim/bert-base-NER"
NER_TOK = AutoTokenizer.from_pretrained(NER_MODEL, use_fast=False)
NER_MDL = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
NER = pipeline("token-classification", model=NER_MDL, tokenizer=NER_TOK,
               aggregation_strategy="simple")

FILE_RE = re.compile(r"\b[\w\-\s]+\.(pdf|csv|xlsx|docx|png|jpg)\b", re.I)
PATH_RE = re.compile(r"(\/[A-Za-z0-9_\- .]+)+")
BOOL_TRUE = re.compile(r"\b(bật|enable|true|đúng|overwrite|ghi đè|yes|on|1)\b", re.I)
BOOL_FALSE= re.compile(r"\b(tắt|disable|false|không|no|off|0)\b", re.I)

def cast_by_type(val: Any, typ: str):
    if val is None: return None
    if typ == "boolean":
        s = str(val).lower()
        if BOOL_TRUE.search(s): return True
        if BOOL_FALSE.search(s): return False
        return None
    if typ == "number":
        try: return float(val)
        except: return None
    if typ == "string": return str(val)
    if typ == "list": return [val]
    if typ == "dictionary": return {"value": val}
    return val

def slot_fill(tpl: TplDoc, text: str) -> Dict[str, Any]:
    ents_raw = NER(text[:512])  # tránh text quá dài
    # ép types JSON-safe
    ents = [
        {
            "entity_group": str(e.get("entity_group")),
            "word": str(e.get("word")),
            "start": int(e.get("start",0)),
            "end": int(e.get("end",0)),
            "score": float(e.get("score",0.0))
        } for e in ents_raw
    ]
    files = FILE_RE.findall(text)
    file_first = None
    if files:
        # FILE_RE với nhóm -> trả ('csv',) ... khớp full cần dùng finditer
        m = re.search(r"\b[\w\-\s]+\.(pdf|csv|xlsx|docx|png|jpg)\b", text, flags=re.I)
        file_first = m.group(0) if m else None

    paths = PATH_RE.findall(text)
    path_first = paths[0] if paths else None

    out: Dict[str, Any] = {}
    for req in tpl.requiredArgs:
        v = None
        if re.search(r"file", req.name, re.I) and file_first:
            v = file_first
        elif re.search(r"folder|path", req.name, re.I) and path_first:
            v = path_first
        elif req.type == "boolean":
            v = cast_by_type(text, "boolean")

        if v is None and ents:
            # lấy một entity làm fallback string
            v = ents[0]["word"]

        out[req.name] = cast_by_type(v, req.type) if v is not None else None

    return out

# ===============================
# 5) Validator & IR builder
# ===============================

def validate_args(tpl: TplDoc, args: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for req in tpl.requiredArgs:
        v = args.get(req.name, None)
        if v in (None, ""):
            errs.append(f"Missing required argument: {req.name} ({req.type})")
            continue
        if req.type == "boolean" and not isinstance(v, bool):
            errs.append(f"Type mismatch at {req.name}: expect boolean")
        if req.type == "number" and not isinstance(v, (int, float)):
            errs.append(f"Type mismatch at {req.name}: expect number")
        if req.type == "string" and not isinstance(v, str):
            errs.append(f"Type mismatch at {req.name}: expect string")
    return errs

def build_task_node(node_id: str, tpl: TplDoc, lane: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return { "id": node_id, "type": "TASK", "name": tpl.templateId, "lane": lane, "args": args }

# ===============================
# 6) Renderer BPMN XML (tối giản, code-first)
# ===============================

# def ir_to_bpmn_xml(ir: Dict[str,Any], keyword_index: Dict[str,str]) -> str:
#     NSMAP = {
#         None: "http://www.omg.org/spec/BPMN/20100524/MODEL",
#         "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
#         "omgdc": "http://www.omg.org/spec/DD/20100524/DC",
#         "omgdi": "http://www.omg.org/spec/DD/20100524/DI",
#         "rpa": "http://example.com/schema/rpa"
#     }
#     defs = etree.Element("definitions", nsmap=NSMAP, id="Defs_1")
#     process = etree.SubElement(defs, "process", id="Process_1", isExecutable="true", name=ir.get("metadata",{}).get("title","Generated"))

#     lane_sets = ir.get("lanes", [])
#     laneSetEl = etree.SubElement(process, "laneSet", id="LaneSet_1") if lane_sets else None
#     laneEls: Dict[str, etree._Element] = {}
#     if lane_sets:
#         for i, ln in enumerate(lane_sets, start=1):
#             laneEl = etree.SubElement(laneSetEl, "lane", id=f"Lane_{i}", name=ln)
#             laneEls[ln] = laneEl

#     nodeEls: Dict[str, etree._Element] = {}
#     start = etree.SubElement(process, "startEvent", id="StartEvent_1", name="Start")
#     nodeEls["START"] = start

#     for n in ir["flow"]:
#         if n["type"] == "END":
#             el = etree.SubElement(process, "endEvent", id=n["id"], name="End")
#         elif n["type"] == "TASK":
#             el = etree.SubElement(process, "serviceTask", id=n["id"], name=n["name"])
#             # extensionElements → rpa:ActivityTemplateRef
#             ext = etree.SubElement(el, "extensionElements")
#             etree.SubElement(ext, "{http://example.com/schema/rpa}ActivityTemplateRef",
#                              templateId=n["name"],
#                              keyword=keyword_index.get(n["name"], ""),
#                              argsJson=json.dumps(n.get("args", {}), ensure_ascii=False))
#         elif n["type"] == "EXCLUSIVE":
#             el = etree.SubElement(process, "exclusiveGateway", id=n["id"], name=n.get("condVar","XOR"))
#         elif n["type"] == "PARALLEL":
#             el = etree.SubElement(process, "parallelGateway", id=n["id"], name="AND")
#         elif n["type"] == "TRIGGER":
#             start.set("name", n.get("name","Start"))
#             el = start
#         else:
#             continue

#         nodeEls[n["id"]] = el
#         ln = n.get("lane")
#         if ln and laneSetEl is not None:
#             # (tối giản) thêm flowNodeRef
#             etree.SubElement(laneEls[ln], "flowNodeRef").text = n["id"]

#     # sequenceFlow
#     for (src, dst, meta) in ir["edges"]:
#         sf = etree.SubElement(process, "sequenceFlow",
#                               id=f"Flow_{src}_{dst}",
#                               sourceRef="StartEvent_1" if src=="START" else src,
#                               targetRef=dst)
#         if meta and meta.get("cond"):
#             cond = etree.SubElement(sf, "conditionExpression")
#             cond.text = meta["cond"]
#         if meta and meta.get("isDefault"):
#             # set default attribute on source gateway (not fully implemented—simple)
#             pass

#     # Auto connect Start → first node if not provided
#     if not any(e[0]=="START" for e in ir["edges"]):
#         first = next((n for n in ir["flow"] if n["type"]!="END"), None)
#         if first:
#             etree.SubElement(process, "sequenceFlow",
#                              id=f"Flow_Start_{first['id']}",
#                              sourceRef="StartEvent_1", targetRef=first["id"])

#     return etree.tostring(defs, pretty_print=True, encoding="utf-8", xml_declaration=True).decode("utf-8")

# ===============================
# 7) Các Node của LangGraph
# ===============================

def node_normalize(state: FlowState) -> FlowState:
    state["normalized"] = normalize_text(state["user_input"])
    return state

def node_retrieve(state: FlowState) -> FlowState:
    cand = retrieve_candidates(state["normalized"], TPL_DOCS, k=5)
    state["candidates"] = cand
    # strategy: pick top-2 có template Drive create/upload nếu có
    picked: List[TplDoc] = []
    ids = [c[0].templateId for c in cand]
    def pick_first(substr: str):
        for t, _ in cand:
            if substr in t.templateId and t not in picked:
                picked.append(t); return
    pick_first("create_folder")
    pick_first("upload_file")
    # nếu thiếu thì cứ chọn top-1/2
    for t,_ in cand:
        if len(picked) >= 2: break
        if t not in picked: picked.append(t)
    state["chosen"] = picked
    return state

def node_slot_fill(state: FlowState) -> FlowState:
    args_map: Dict[str, Dict[str, Any]] = {}
    for tpl in state["chosen"]:
        args_map[tpl.templateId] = slot_fill(tpl, state["user_input"])
    state["args_map"] = args_map
    return state

def node_validate(state: FlowState) -> FlowState:
    errors: List[str] = []
    # ép/điền default hợp lý (example)
    if "drive.create_folder" in state["args_map"]:
        a = state["args_map"]["drive.create_folder"]
        a.setdefault("Folder name", "Invoices")
        a.setdefault("Parent Folder Path", "/Shared")
    if "drive.upload_file" in state["args_map"]:
        a = state["args_map"]["drive.upload_file"]
        a["Overwrite"]   = True if a.get("Overwrite")   is None else a["Overwrite"]
        a["Make Folder"] = False if a.get("Make Folder") is None else a["Make Folder"]
        a.setdefault("Folder Path", "/Shared/Invoices")
        a.setdefault("File name", "aug.csv")

    for tpl in state["chosen"]:
        errs = validate_args(tpl, state["args_map"][tpl.templateId])
        errors.extend(errs)

    state["errors"] = errors
    return state

def node_build_ir(state: FlowState) -> FlowState:
    if state.get("errors"): return state
    lanes = list({ "Drive" })  # demo: 1 lane
    flow = [{"id":"t1","type":"TRIGGER","name":"Manual","lane":"Drive"}]
    edges = [("START","t1",{})]

    nid = 2
    for tpl in state["chosen"]:
        flow.append(build_task_node(f"t{nid}", tpl, "Drive", state["args_map"][tpl.templateId]))
        edges.append((f"t{nid-1}" if nid>2 else "t1", f"t{nid}", {}))
        nid += 1
    flow.append({"id":"end","type":"END","lane":"Drive"})
    edges.append((f"t{nid-1}", "end", {}))

    ir = {
        "metadata": { "title": "Drive Flow (LangGraph Demo)" },
        "lanes": lanes,
        "flow": flow,
        "edges": edges
    }
    state["ir"] = ir
    return state

# def node_render(state: FlowState) -> FlowState:
#     if state.get("errors"): return state
#     act_index = { d.templateId: d.keyword for d in TPL_DOCS }
#     state["bpmn_xml"] = ir_to_bpmn_xml(state["ir"], act_index)
#     return state

# ===============================
# 8) Xây đồ thị LangGraph
# ===============================

def build_graph():
    g = StateGraph(FlowState)

    g.add_node("normalize", node_normalize)
    g.add_node("retrieve", node_retrieve)
    g.add_node("slot_fill", node_slot_fill)
    g.add_node("validate", node_validate)
    g.add_node("build_ir", node_build_ir)
    # g.add_node("render", node_render)

    g.set_entry_point("normalize")
    g.add_edge("normalize", "retrieve")
    g.add_edge("retrieve", "slot_fill")
    g.add_edge("slot_fill", "validate")

    # rẽ nhánh nếu lỗi
    def has_errors(state: FlowState):
        return "render" if not state.get("errors") else END
    g.add_conditional_edges("validate", has_errors, path_map={"render":"render", END: END})

    g.add_edge("render", END)
    return g.compile()

# ===============================
# 9) Chạy demo
# ===============================

if __name__ == "__main__":
    user_text = "Tạo thư mục 'Invoices' trong Google Drive dưới /Shared rồi tải file aug.csv vào đó, cho phép ghi đè."
    app = build_graph()
    try:
        display(Image(app.get_graph().draw_mermaid_png()))
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)
        print("Saved graph.png")
    except Exception:
        pass
    # out: FlowState = app.invoke({"user_input": user_text})

    # print("=== Top-k candidates ===")
    # for i,(tpl,score) in enumerate(out.get("candidates", [])[:5], start=1):
    #     print(f"{i}. {tpl.templateId}  score={score:.3f}")

    # if out.get("errors"):
    #     print("\nERRORS:")
    #     for e in out["errors"]:
    #         print(" -", e)
    # else:
    #     print("\nIR:\n", json.dumps(out["ir"], ensure_ascii=False, indent=2))
    #     print("\nBPMN XML (first 40 lines):")
    #     xml_lines = out["bpmn_xml"].splitlines()
    #     print("\n".join(xml_lines[:40]))
    #     with open("flow_langgraph.bpmn","w",encoding="utf-8") as f:
    #         f.write(out["bpmn_xml"])
    #     print("\n✅ Saved: flow_langgraph.bpmn")
