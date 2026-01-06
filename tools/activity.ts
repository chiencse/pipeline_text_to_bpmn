// activity.ts
export type ArgSpec = { name: string; type: string; keywordArg?: string };
export type TplDoc = {
  templateId: string;
  pkg: string;
  text: string;
  keyword: string;
  requiredArgs: ArgSpec[];
};
export enum RFVarType {
  "scalar" = "$",
  "any" = "$",
  "dictionary" = "&",
  "list" = "@",
}

export function buildTplDocs(): TplDoc[] {
  return ActivityPackages.flatMap((pkg) =>
    pkg.activityTemplates.map((t: any) => ({
      templateId: t.templateId,
      pkg: pkg._id,
      keyword: t.keyword,
      text: [
        pkg.displayName,
        t.displayName,
        t.description,
        t.keyword,
        ...Object.entries(t.arguments ?? {}).map(
          ([k, v]: any) => `${k} ${v.type} ${v.keywordArg ?? ""}`
        ),
      ]
        .join(" ")
        .toLowerCase(),
      requiredArgs: Object.entries(t.arguments ?? {}).map(([k, v]: any) => ({
        name: k,
        type: v.type,
        keywordArg: v.keywordArg,
      })),
    }))
  );
}

export const ActivityPackages = [
  // Drive
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
  // Gmail
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
  //Sheets
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
  // Classroom
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
  // Form
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
            // keywordArg: 'list_',
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
        keyword: "Upload Drive File",
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
  {
    _id: "rpa-erpnext",
    displayName: "ERPNext",
    description:
      "Automate procurement workflows in ERPNext (Item, MR, RFQ, PO)",
    library: "RPA.ERPNext",
    activityTemplates: [
      {
        templateId: "connect_to_erpnext",
        displayName: "Connect to ERPNext",
        description: "Connect to ERPNext using a base URL and API token",
        iconCode: "FaLink",
        type: "activity",
        keyword: "Connect To ERPNext",
        arguments: {
          "Base URL": {
            type: "string",
            description: "The base URL of the ERPNext system",
            keywordArg: "base_url",
            value: "",
          },
          "Token File Path": {
            type: "connection.ERPNext",
            description:
              "The path to the file containing the ERPNext access token",
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
          description: "Indicates successful connection to ERPNext",
        },
      },
      {
        templateId: "ensure_item_exists",
        displayName: "Ensure Item Exists",
        description:
          "Check if an Item exists in ERPNext; if not, create one automatically.",
        iconCode: "FaBox",
        type: "activity",
        keyword: "Ensure ERP Item Exists",
        arguments: {
          "Item Code": {
            type: "string",
            description: "The code of the item (unique)",
            keywordArg: "item_code",
            value: "",
          },
          "Item Name": {
            type: "string",
            description: "The display name of the item",
            keywordArg: "item_name",
            value: "",
          },
          "Unit of Measure": {
            type: "string",
            description: "The default stock unit (e.g., Nos, Kg)",
            keywordArg: "uom",
            value: "Nos",
          },
        },
        return: {
          displayName: "Item Name",
          type: "string",
          description: "The name of the created or existing item",
        },
      },
      {
        templateId: "create_material_request",
        displayName: "Create Material Request",
        description:
          "Create a Material Request document in ERPNext with multiple items.",
        iconCode: "FaClipboardList",
        type: "activity",
        keyword: "Create ERP Material Request",
        arguments: {
          Items: {
            type: "list",
            description:
              "List of items to request. Each item must include item_code, qty, schedule_date",
            keywordArg: "items",
            value: [],
          },
        },
        return: {
          displayName: "Material Request ID",
          type: "string",
          description: "The name of the created Material Request document",
        },
      },
      {
        templateId: "create_request_for_quotation",
        displayName: "Create Request For Quotation (RFQ)",
        description: "Create an RFQ in ERPNext for one or more suppliers.",
        iconCode: "FaFileInvoiceDollar",
        type: "activity",
        keyword: "Create ERP RFQ",
        arguments: {
          Items: {
            type: "list",
            description: "List of items to request for quotation",
            keywordArg: "items",
            value: [],
          },
          Suppliers: {
            type: "list",
            description: "List of supplier names",
            keywordArg: "suppliers",
            value: [],
          },
          "Message For Supplier": {
            type: "string",
            description: "Message shown to suppliers in the RFQ",
            keywordArg: "message_for_supplier",
            value: "Xin vui lòng báo giá cho các mặt hàng này.",
          },
        },
        return: {
          displayName: "RFQ ID",
          type: "string",
          description: "The name of the created RFQ document",
        },
      },
      {
        templateId: "create_purchase_order",
        displayName: "Create Purchase Order",
        description:
          "Create a Purchase Order document in ERPNext for a supplier.",
        iconCode: "FaFileInvoice",
        type: "activity",
        keyword: "Create ERP Purchase Order",
        arguments: {
          Supplier: {
            type: "string",
            description: "Supplier name",
            keywordArg: "supplier",
            value: "",
          },
          Items: {
            type: "list",
            description: "List of items to order (with item_code, qty, rate)",
            keywordArg: "items",
            value: [],
          },
          "Default Rate": {
            type: "number",
            description: "Default rate per item if not specified",
            keywordArg: "default_rate",
            value: 1000,
          },
        },
        return: {
          displayName: "Purchase Order ID",
          type: "string",
          description: "The name of the created Purchase Order",
        },
      },
      {
        templateId: "submit_erp_document",
        displayName: "Submit ERP Document",
        description:
          "Submit an existing ERPNext document (Material Request, RFQ, or Purchase Order).",
        iconCode: "FaCheckCircle",
        type: "activity",
        keyword: "Submit ERP Document",
        arguments: {
          DocType: {
            type: "string",
            description:
              "The ERPNext doctype (e.g., Material Request, Request for Quotation, Purchase Order)",
            keywordArg: "doctype",
            value: "",
          },
          "Document ID": {
            type: "string",
            description: "The name or ID of the document to submit",
            keywordArg: "name",
            value: "",
          },
        },
        return: {
          displayName: "Submission Result",
          type: "string",
          description: "The name of the submitted document",
        },
      },
    ],
  },
  // Moodle
  {
    _id: "moodle",
    displayName: "Moodle",
    description: "Integrate with Moodle LMS for course and quiz management",
    library: "RPA.Moodle",
    activityTemplates: [
      // Connection Setup
      {
        templateId: "moodle.setup_connection",
        displayName: "Setup Moodle Connection",
        description: "Set up Moodle connection for following tasks",
        iconCode: "FaGraduationCap",
        type: "activity",
        keyword: "Set Up Moodle Connection",
        arguments: {
          Connection: {
            type: "connection.Moodle",
            keywordArg: "token_file_path",
            provider: "Moodle",
            description: "Your connection with Moodle",
            value: null,
          },
        },
      },
      {
        templateId: "moodle.setup_google_connection",
        displayName: "Setup Google Docs Connection",
        description: "Setup Google API connection for Google Docs integration",
        iconCode: "FaGoogle",
        type: "activity",
        keyword: "Setup Google Connection",
        arguments: {
          Connection: {
            type: "connection.Google Drive",
            keywordArg: "token_file_path",
            description: "Path to Google API token JSON file",
            value: "",
          },
        },
      },
      // Course Category Management
      {
        templateId: "moodle.get_course_categories",
        displayName: "Get Course Categories",
        description: "Get list of all course categories",
        iconCode: "FaFolderOpen",
        type: "activity",
        keyword: "Get Course Categories",
        arguments: {},
        return: {
          displayName: "Categories",
          type: "list",
          description: "List of course categories",
        },
      },
      {
        templateId: "moodle.create_course_category",
        displayName: "Create Course Category",
        description: "Create a new course category",
        iconCode: "FaFolderPlus",
        type: "activity",
        keyword: "Create Course Category",
        arguments: {
          Name: {
            type: "string",
            keywordArg: "name",
            description: "Category name",
            value: "",
          },
          Parent: {
            type: "number",
            keywordArg: "parent",
            description: "Parent category ID (0 for top level)",
            value: 0,
          },
          Description: {
            type: "string",
            keywordArg: "description",
            description: "Category description",
            value: "",
          },
        },
        return: {
          displayName: "Category",
          type: "dictionary",
          description: "Created category information",
        },
      },
      {
        templateId: "moodle.ensure_category_exists",
        displayName: "Ensure Category Exists",
        description: "Ensure category exists, create if not found",
        iconCode: "FaCheckCircle",
        type: "activity",
        keyword: "Ensure Category Exists",
        arguments: {
          Name: {
            type: "string",
            keywordArg: "name",
            description: "Category name",
            value: "",
          },
          Parent: {
            type: "number",
            keywordArg: "parent",
            description: "Parent category ID",
            value: 0,
          },
        },
        return: {
          displayName: "Category",
          type: "dictionary",
          description: "Category information",
        },
      },
      // Course Management
      {
        templateId: "moodle.create_course",
        displayName: "Create Course",
        description: "Create a new course in Moodle",
        iconCode: "FaBook",
        type: "activity",
        keyword: "Create Course",
        arguments: {
          Fullname: {
            type: "string",
            keywordArg: "fullname",
            description: "Full name of the course",
            value: "",
          },
          Shortname: {
            type: "string",
            keywordArg: "shortname",
            description: "Short name (unique identifier)",
            value: "",
          },
          "Category ID": {
            type: "number",
            keywordArg: "categoryid",
            description: "Category ID",
            value: 1,
          },
          Summary: {
            type: "string",
            keywordArg: "summary",
            description: "Course summary/description",
            value: "",
          },
          Format: {
            type: "string",
            keywordArg: "format",
            description: "Course format (topics, weeks, social)",
            value: "topics",
          },
        },
        return: {
          displayName: "Course",
          type: "dictionary",
          description: "Created course information",
        },
      },
      {
        templateId: "moodle.get_course_by_shortname",
        displayName: "Get Course By Shortname",
        description: "Get course information by shortname",
        iconCode: "FaSearch",
        type: "activity",
        keyword: "Get Course By Shortname",
        arguments: {
          Shortname: {
            type: "string",
            keywordArg: "shortname",
            description: "Course shortname",
            value: "",
          },
        },
        return: {
          displayName: "Course",
          type: "dictionary",
          description: "Course information",
        },
      },
      {
        templateId: "moodle.ensure_course_exists",
        displayName: "Ensure Course Exists",
        description: "Ensure course exists, create if not found",
        iconCode: "FaCheckCircle",
        type: "activity",
        keyword: "Ensure Course Exists",
        arguments: {
          Fullname: {
            type: "string",
            keywordArg: "fullname",
            description: "Full name of the course",
            value: "",
          },
          Shortname: {
            type: "string",
            keywordArg: "shortname",
            description: "Short name (unique)",
            value: "",
          },
          "Category ID": {
            type: "number",
            keywordArg: "categoryid",
            description: "Category ID",
            value: 1,
          },
          Summary: {
            type: "string",
            keywordArg: "summary",
            description: "Course summary",
            value: "",
          },
        },
        return: {
          displayName: "Course",
          type: "dictionary",
          description: "Course information",
        },
      },
      {
        templateId: "moodle.get_course_contents",
        displayName: "Get Course Contents",
        description: "Get contents of a course",
        iconCode: "FaListAlt",
        type: "activity",
        keyword: "Get Course Contents",
        arguments: {
          "Course ID": {
            type: "number",
            keywordArg: "courseid",
            description: "Course ID",
            value: 0,
          },
        },
        return: {
          displayName: "Contents",
          type: "list",
          description: "List of course contents",
        },
      },
      // Quiz Management
      {
        templateId: "moodle.generate_quiz_gift_file",
        displayName: "Generate Quiz GIFT File",
        description: "Generate GIFT format file for Moodle quiz import",
        iconCode: "FaFileAlt",
        type: "activity",
        keyword: "Generate Quiz GIFT File",
        arguments: {
          Questions: {
            type: "list",
            keywordArg: "questions",
            description: "List of question dictionaries",
            value: [],
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save GIFT file",
            value: "",
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to generated GIFT file",
        },
      },
      {
        templateId: "moodle.generate_quiz_xml_file",
        displayName: "Generate Quiz XML File",
        description: "Generate Moodle XML format file for quiz import",
        iconCode: "FaFileCode",
        type: "activity",
        keyword: "Generate Quiz XML File",
        arguments: {
          Questions: {
            type: "list",
            keywordArg: "questions",
            description: "List of question dictionaries",
            value: [],
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save XML file",
            value: "",
          },
          "Quiz Name": {
            type: "string",
            keywordArg: "quiz_name",
            description: "Name of the quiz",
            value: "Quiz",
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to generated XML file",
        },
      },
      // User Management
      {
        templateId: "moodle.create_user",
        displayName: "Create User",
        description: "Create a new user in Moodle",
        iconCode: "FaUserPlus",
        type: "activity",
        keyword: "Create User",
        arguments: {
          Username: {
            type: "string",
            keywordArg: "username",
            description: "Username",
            value: "",
          },
          Password: {
            type: "string",
            keywordArg: "password",
            description: "Password",
            value: "",
          },
          Firstname: {
            type: "string",
            keywordArg: "firstname",
            description: "First name",
            value: "",
          },
          Lastname: {
            type: "string",
            keywordArg: "lastname",
            description: "Last name",
            value: "",
          },
          Email: {
            type: "email",
            keywordArg: "email",
            description: "Email address",
            value: "",
          },
        },
        return: {
          displayName: "User",
          type: "dictionary",
          description: "Created user information",
        },
      },
      {
        templateId: "moodle.enroll_user_in_course",
        displayName: "Enroll User In Course",
        description: "Enroll a user in a course",
        iconCode: "FaUserGraduate",
        type: "activity",
        keyword: "Enroll User In Course",
        arguments: {
          "User ID": {
            type: "number",
            keywordArg: "userid",
            description: "User ID",
            value: 0,
          },
          "Course ID": {
            type: "number",
            keywordArg: "courseid",
            description: "Course ID",
            value: 0,
          },
          "Role ID": {
            type: "number",
            keywordArg: "roleid",
            description: "Role ID (5=Student, 3=Teacher)",
            value: 5,
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Enrollment result",
        },
      },
      {
        templateId: "moodle.enrol_user",
        displayName: "Enrol User",
        description: "Enrol user by username and course shortname",
        iconCode: "FaUserCheck",
        type: "activity",
        keyword: "Enrol User",
        arguments: {
          Username: {
            type: "string",
            keywordArg: "username",
            description: "Username",
            value: "",
          },
          "Course Shortname": {
            type: "string",
            keywordArg: "course_shortname",
            description: "Course shortname",
            value: "",
          },
          "Role ID": {
            type: "number",
            keywordArg: "roleid",
            description: "Role ID (5=Student, 3=Teacher)",
            value: 5,
          },
        },
        return: {
          displayName: "Result",
          type: "dictionary",
          description: "Enrollment result",
        },
      },
      // Google Docs Integration
      {
        templateId: "moodle.get_drive_file_id_from_url",
        displayName: "Get Google Drive File ID From URL",
        description: "Extract file ID from Google Drive/Sheets/Docs URL",
        iconCode: "FaLink",
        type: "activity",
        keyword: "Get Google Drive File ID From URL",
        arguments: {
          URL: {
            type: "string",
            keywordArg: "url",
            description: "Google Drive/Sheets/Docs URL",
            value: "",
          },
        },
        return: {
          displayName: "File ID",
          type: "string",
          description: "Google Drive file ID",
        },
      },
      {
        templateId: "moodle.read_google_doc_content",
        displayName: "Read Google Doc Content",
        description: "Read text content from Google Docs",
        iconCode: "FaFileAlt",
        type: "activity",
        keyword: "Read Google Doc Content",
        arguments: {
          "Doc ID": {
            type: "string",
            keywordArg: "doc_id",
            description: "Google Doc ID",
            value: "",
          },
        },
        return: {
          displayName: "Content",
          type: "string",
          description: "Document text content",
        },
      },
      {
        templateId: "moodle.read_quiz_from_google_doc",
        displayName: "Read Quiz From Google Doc",
        description: "Parse quiz questions and answers from Google Doc",
        iconCode: "FaQuestionCircle",
        type: "activity",
        keyword: "Read Quiz From Google Doc",
        arguments: {
          "Doc ID": {
            type: "string",
            keywordArg: "doc_id",
            description: "Google Doc ID",
            value: "",
          },
          Delimiter: {
            type: "string",
            keywordArg: "delimiter",
            description: "Delimiter between questions and answers",
            value: "---HẾT---",
          },
        },
        return: {
          displayName: "Questions",
          type: "list",
          description: "List of parsed questions with answers",
        },
      },
      {
        templateId: "moodle.create_quiz_from_google_doc",
        displayName: "Create Quiz From Google Doc",
        description: "Create quiz file from Google Doc",
        iconCode: "FaFileImport",
        type: "activity",
        keyword: "Create Quiz From Google Doc",
        arguments: {
          "Doc ID": {
            type: "string",
            keywordArg: "doc_id",
            description: "Google Doc ID or URL",
            value: "",
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save quiz file",
            value: "",
          },
          "Output Format": {
            type: "string",
            keywordArg: "output_format",
            description: "Output format (gift or xml)",
            value: "gift",
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to generated quiz file",
        },
      },
      // File Upload
      {
        templateId: "moodle.upload_file_to_moodle",
        displayName: "Upload File To Moodle",
        description: "Upload a file to Moodle server",
        iconCode: "FaUpload",
        type: "activity",
        keyword: "Upload File To Moodle",
        arguments: {
          "File Path": {
            type: "string",
            keywordArg: "file_path",
            description: "Path to file to upload",
            value: "",
          },
          "Context ID": {
            type: "number",
            keywordArg: "contextid",
            description: "Context ID (default: 1 for system)",
            value: 1,
          },
        },
        return: {
          displayName: "File Info",
          type: "dictionary",
          description: "Uploaded file information",
        },
      },
      // Bulk Operations
      {
        templateId: "moodle.parse_students_from_excel",
        displayName: "Parse Students From Excel",
        description: "Parse student list from Excel file",
        iconCode: "FaFileExcel",
        type: "activity",
        keyword: "Parse Students From Excel",
        arguments: {
          "File Path": {
            type: "string",
            keywordArg: "file_path",
            description: "Path to Excel file",
            value: "",
          },
        },
        return: {
          displayName: "Students",
          type: "list",
          description: "List of student dictionaries",
        },
      },
      {
        templateId: "moodle.parse_courses_from_excel",
        displayName: "Parse Courses From Excel",
        description: "Parse course list from Excel file",
        iconCode: "FaFileExcel",
        type: "activity",
        keyword: "Parse Courses From Excel",
        arguments: {
          "File Path": {
            type: "string",
            keywordArg: "file_path",
            description: "Path to Excel file",
            value: "",
          },
        },
        return: {
          displayName: "Courses",
          type: "list",
          description: "List of course dictionaries",
        },
      },
      {
        templateId: "moodle.bulk_create_students",
        displayName: "Bulk Create Students",
        description: "Create multiple students at once",
        iconCode: "FaUsers",
        type: "activity",
        keyword: "Bulk Create Students",
        arguments: {
          Students: {
            type: "list",
            keywordArg: "students",
            description: "List of student dictionaries",
            value: [],
          },
        },
        return: {
          displayName: "Summary",
          type: "dictionary",
          description: "Creation summary with success/failure counts",
        },
      },
      {
        templateId: "moodle.bulk_create_courses_and_enrol_students",
        displayName: "Bulk Create Courses And Enrol Students",
        description: "Create multiple courses and enrol students",
        iconCode: "FaChalkboardTeacher",
        type: "activity",
        keyword: "Bulk Create Courses And Enrol Students",
        arguments: {
          Courses: {
            type: "list",
            keywordArg: "courses",
            description: "List of course dictionaries",
            value: [],
          },
          "Student Usernames": {
            type: "list",
            keywordArg: "student_usernames",
            description: "List of student usernames to enrol",
            value: [],
          },
        },
        return: {
          displayName: "Summary",
          type: "dictionary",
          description: "Creation and enrollment summary",
        },
      },
      // Google Drive Integration
      {
        templateId: "moodle.upload_file_to_google_drive",
        displayName: "Upload File To Google Drive",
        description: "Upload file to Google Drive",
        iconCode: "FaGoogleDrive",
        type: "activity",
        keyword: "Upload File To Google Drive",
        arguments: {
          "File Path": {
            type: "string",
            keywordArg: "file_path",
            description: "Path to file to upload",
            value: "",
          },
          "Folder ID": {
            type: "string",
            keywordArg: "folder_id",
            description: "Google Drive folder ID (optional)",
            value: "",
          },
        },
        return: {
          displayName: "File Info",
          type: "dictionary",
          description: "Uploaded file info with ID and links",
        },
      },
      {
        templateId: "moodle.download_file_from_google_drive",
        displayName: "Download File From Google Drive",
        description: "Download file from Google Drive",
        iconCode: "FaDownload",
        type: "activity",
        keyword: "Download File From Google Drive",
        arguments: {
          "File ID": {
            type: "string",
            keywordArg: "file_id",
            description: "Google Drive file ID",
            value: "",
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save downloaded file",
            value: "",
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to downloaded file",
        },
      },
      {
        templateId: "moodle.create_quiz_and_upload_to_drive",
        displayName: "Create Quiz And Upload To Drive",
        description: "Create quiz from Google Doc and upload to Drive",
        iconCode: "FaCloudUploadAlt",
        type: "activity",
        keyword: "Create Quiz And Upload To Drive",
        arguments: {
          "Doc ID": {
            type: "string",
            keywordArg: "doc_id",
            description: "Google Doc ID with questions",
            value: "",
          },
          "Folder ID": {
            type: "string",
            keywordArg: "folder_id",
            description: "Google Drive folder ID (optional)",
            value: "",
          },
        },
        return: {
          displayName: "Upload Info",
          type: "dictionary",
          description: "Upload information with links",
        },
      },
      // User Lookup
      {
        templateId: "moodle.get_user_by_username",
        displayName: "Get User By Username",
        description: "Get user information by username",
        iconCode: "FaUserSearch",
        type: "activity",
        keyword: "Get User By Username",
        arguments: {
          Username: {
            type: "string",
            keywordArg: "username",
            description: "Username to search for",
            value: "",
          },
        },
        return: {
          displayName: "User Info",
          type: "dictionary",
          description: "User information dictionary",
        },
      },
      {
        templateId: "moodle.get_user_by_email",
        displayName: "Get User By Email",
        description: "Get user information by email address",
        iconCode: "FaEnvelope",
        type: "activity",
        keyword: "Get User By Email",
        arguments: {
          Email: {
            type: "email",
            keywordArg: "email",
            description: "Email address to search for",
            value: "",
          },
        },
        return: {
          displayName: "User Info",
          type: "dictionary",
          description: "User information dictionary",
        },
      },
      {
        templateId: "moodle.get_user_by_username_or_email",
        displayName: "Get User By Username Or Email",
        description: "Get user by username or email (flexible search)",
        iconCode: "FaSearch",
        type: "activity",
        keyword: "Get User By Username Or Email",
        arguments: {
          Identifier: {
            type: "string",
            keywordArg: "identifier",
            description: "Username or email address",
            value: "",
          },
        },
        return: {
          displayName: "User Info",
          type: "dictionary",
          description: "User information dictionary",
        },
      },
      // Assignment Management
      {
        templateId: "moodle.get_assignment_id_from_url",
        displayName: "Get Assignment ID From URL",
        description: "Extract assignment course module ID from Moodle URL",
        iconCode: "FaLink",
        type: "activity",
        keyword: "Get Assignment ID From URL",
        arguments: {
          URL: {
            type: "string",
            keywordArg: "url",
            description: "Moodle assignment URL",
            value: "",
          },
        },
        return: {
          displayName: "Assignment ID",
          type: "number",
          description: "Course module ID (cmid)",
        },
      },
      {
        templateId: "moodle.get_assignment_instance_id_from_cmid",
        displayName: "Get Assignment Instance ID From Course Module ID",
        description: "Convert course module ID to assignment instance ID",
        iconCode: "FaExchangeAlt",
        type: "activity",
        keyword: "Get Assignment Instance ID From Course Module ID",
        arguments: {
          "Course ID": {
            type: "number",
            keywordArg: "course_id",
            description: "Moodle course ID",
            value: 0,
          },
          "Course Module ID": {
            type: "number",
            keywordArg: "cmid",
            description: "Course module ID from URL",
            value: 0,
          },
        },
        return: {
          displayName: "Instance ID",
          type: "number",
          description: "Assignment instance ID",
        },
      },
      {
        templateId: "moodle.create_submission_for_student",
        displayName: "Create Submission For Student",
        description: "Create/submit assignment for a student",
        iconCode: "FaFileUpload",
        type: "activity",
        keyword: "Create Submission For Student",
        arguments: {
          "Assignment ID": {
            type: "number",
            keywordArg: "assignment_id",
            description: "Assignment instance ID",
            value: 0,
          },
          "User ID": {
            type: "number",
            keywordArg: "user_id",
            description: "Student user ID",
            value: 0,
          },
          "Submission Text": {
            type: "string",
            keywordArg: "submission_text",
            description: "Text content for submission",
            value: "Auto-submitted for grading",
          },
        },
        return: {
          displayName: "Submission Result",
          type: "dictionary",
          description: "Submission creation result",
        },
      },
      {
        templateId: "moodle.submit_assignment_for_students",
        displayName: "Submit Assignment For Students",
        description: "Create submissions for multiple students",
        iconCode: "FaUsersCog",
        type: "activity",
        keyword: "Submit Assignment For Students",
        arguments: {
          "Assignment ID": {
            type: "number",
            keywordArg: "assignment_id",
            description: "Assignment instance ID",
            value: 0,
          },
          "Student Usernames": {
            type: "list",
            keywordArg: "student_usernames",
            description: "List of student usernames",
            value: [],
          },
        },
        return: {
          displayName: "Summary",
          type: "dictionary",
          description: "Submission creation summary",
        },
      },
      // AI Grading with Gemini
      {
        templateId: "moodle.setup_gemini_ai",
        displayName: "Setup Gemini AI",
        description: "Setup Gemini AI for automatic grading",
        iconCode: "FaRobot",
        type: "activity",
        keyword: "Setup Gemini AI",
        arguments: {
          "API Key": {
            type: "string",
            keywordArg: "api_key",
            description: "Gemini API key from Google AI Studio",
            value: "",
          },
        },
      },
      {
        templateId: "moodle.grade_submission_with_gemini",
        displayName: "Grade Submission With Gemini",
        description: "Grade a student submission using Gemini AI",
        iconCode: "FaCheckSquare",
        type: "activity",
        keyword: "Grade Submission With Gemini",
        arguments: {
          "Submission Text": {
            type: "string",
            keywordArg: "submission_text",
            description: "Student's submission text",
            value: "",
          },
          "Question Text": {
            type: "string",
            keywordArg: "question_text",
            description: "Test questions",
            value: "",
          },
          "Answer Key": {
            type: "string",
            keywordArg: "answer_key",
            description: "Answer key",
            value: "",
          },
          "Max Score": {
            type: "number",
            keywordArg: "max_score",
            description: "Maximum score",
            value: 10,
          },
        },
        return: {
          displayName: "Grading Result",
          type: "dictionary",
          description: "Score, feedback, and details",
        },
      },
      {
        templateId: "moodle.grade_image_submission_with_gemini_vision",
        displayName: "Grade Image Submission With Gemini Vision",
        description: "Grade student submission from image using Gemini Vision",
        iconCode: "FaImage",
        type: "activity",
        keyword: "Grade Image Submission With Gemini Vision",
        arguments: {
          "Image Path": {
            type: "string",
            keywordArg: "image_path",
            description: "Path to student's answer sheet image",
            value: "",
          },
          Questions: {
            type: "string",
            keywordArg: "questions",
            description: "Questions text",
            value: "",
          },
          "Answer Key": {
            type: "string",
            keywordArg: "answer_key",
            description: "Answer key text",
            value: "",
          },
          "Max Score": {
            type: "number",
            keywordArg: "max_score",
            description: "Maximum score",
            value: 10.0,
          },
        },
        return: {
          displayName: "Grading Result",
          type: "dictionary",
          description: "Score and feedback from image analysis",
        },
      },
      // Grade Upload and Export
      {
        templateId: "moodle.upload_grades_to_moodle",
        displayName: "Upload Grades To Moodle",
        description: "Upload grades to Moodle assignment gradebook",
        iconCode: "FaCloudUploadAlt",
        type: "activity",
        keyword: "Upload Grades To Moodle",
        arguments: {
          "Assignment ID": {
            type: "number",
            keywordArg: "assignment_id",
            description: "Moodle assignment instance ID",
            value: 0,
          },
          Grades: {
            type: "list",
            keywordArg: "grades",
            description:
              "List of grade dictionaries with student_name, score, feedback",
            value: [],
          },
        },
        return: {
          displayName: "Upload Summary",
          type: "dictionary",
          description: "Summary of uploaded grades",
        },
      },
      {
        templateId: "moodle.export_grading_results_to_excel",
        displayName: "Export Grading Results To Excel",
        description: "Export grading results to Excel file",
        iconCode: "FaFileExcel",
        type: "activity",
        keyword: "Export Grading Results To Excel",
        arguments: {
          "Grading Results": {
            type: "list",
            keywordArg: "grading_results",
            description: "List of grading result dictionaries",
            value: [],
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save Excel file",
            value: "",
          },
          "Max Score": {
            type: "number",
            keywordArg: "max_score",
            description: "Maximum score for percentage calculation",
            value: 10.0,
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to created Excel file",
        },
      },
      {
        templateId: "moodle.export_grading_results_to_csv",
        displayName: "Export Grading Results To CSV",
        description:
          "Export grading results to CSV for Moodle gradebook import",
        iconCode: "FaFileCsv",
        type: "activity",
        keyword: "Export Grading Results To CSV",
        arguments: {
          "Grading Results": {
            type: "list",
            keywordArg: "grading_results",
            description: "List of grading result dictionaries",
            value: [],
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save CSV file",
            value: "",
          },
          "Max Score": {
            type: "number",
            keywordArg: "max_score",
            description: "Maximum score",
            value: 10.0,
          },
          "Grade Item Name": {
            type: "string",
            keywordArg: "grade_item_name",
            description: "Name of the grade item in Moodle",
            value: "Auto Grading",
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to created CSV file",
        },
      },
      {
        templateId: "moodle.export_detailed_grading_results_to_excel",
        displayName: "Export Detailed Grading Results To Excel",
        description:
          "Export detailed student answers and grading breakdown to Excel",
        iconCode: "FaFileExcel",
        type: "activity",
        keyword: "Export Detailed Grading Results To Excel",
        arguments: {
          "Grading Results": {
            type: "list",
            keywordArg: "grading_results",
            description: "List of grading result dictionaries",
            value: [],
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save Excel file",
            value: "",
          },
          "Max Score": {
            type: "number",
            keywordArg: "max_score",
            description: "Maximum score for percentage calculation",
            value: 10.0,
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to created Excel file with detailed results",
        },
      },
      {
        templateId: "moodle.export_student_answers_to_csv",
        displayName: "Export Student Answers To CSV",
        description:
          "Export only student answers to CSV (without grading details)",
        iconCode: "FaFileCsv",
        type: "activity",
        keyword: "Export Student Answers To CSV",
        arguments: {
          "Grading Results": {
            type: "list",
            keywordArg: "grading_results",
            description: "List of grading result dictionaries",
            value: [],
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save CSV file",
            value: "",
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to created CSV file with student answers",
        },
      },
      // Complete Workflows
      {
        templateId: "moodle.complete_auto_grading_and_upload_workflow",
        displayName: "Complete Auto Grading And Upload Workflow",
        description: "Grade submissions and upload results to Moodle",
        iconCode: "FaProjectDiagram",
        type: "activity",
        keyword: "Complete Auto Grading And Upload Workflow",
        arguments: {
          "Submissions Folder ID": {
            type: "string",
            keywordArg: "submissions_folder_id",
            description: "Google Drive folder with submissions",
            value: "",
          },
          "Question File ID": {
            type: "string",
            keywordArg: "question_file_id",
            description: "File ID of questions",
            value: "",
          },
          "Answer Key File ID": {
            type: "string",
            keywordArg: "answer_key_file_id",
            description: "File ID of answer key",
            value: "",
          },
          "Course ID": {
            type: "number",
            keywordArg: "course_id",
            description: "Moodle course ID",
            value: 0,
          },
          "Assignment ID": {
            type: "number",
            keywordArg: "assignment_id",
            description: "Moodle assignment ID",
            value: 0,
          },
          "Max Score": {
            type: "number",
            keywordArg: "max_score",
            description: "Maximum score",
            value: 10.0,
          },
        },
        return: {
          displayName: "Complete Results",
          type: "dictionary",
          description: "Grading and upload results",
        },
      },
      {
        templateId: "moodle.grade_all_image_submissions",
        displayName: "Grade All Image Submissions",
        description: "Grade all image submissions using Gemini Vision",
        iconCode: "FaImages",
        type: "activity",
        keyword: "Grade All Image Submissions",
        arguments: {
          "Submissions Folder ID": {
            type: "string",
            keywordArg: "submissions_folder_id",
            description: "Google Drive folder with image submissions",
            value: "",
          },
          "Question File ID": {
            type: "string",
            keywordArg: "question_file_id",
            description: "File ID of questions",
            value: "",
          },
          "Answer Key File ID": {
            type: "string",
            keywordArg: "answer_key_file_id",
            description: "File ID of answer key",
            value: "",
          },
          "Max Score": {
            type: "number",
            keywordArg: "max_score",
            description: "Maximum score",
            value: 10.0,
          },
        },
        return: {
          displayName: "Grades List",
          type: "list",
          description: "List of grades",
        },
      },
      {
        templateId: "moodle.list_files_in_google_drive_folder",
        displayName: "List Files In Google Drive Folder",
        description: "List all files in a Google Drive folder",
        iconCode: "FaFolderOpen",
        type: "activity",
        keyword: "List Files In Google Drive Folder",
        arguments: {
          "Folder ID": {
            type: "string",
            keywordArg: "folder_id",
            description: "Google Drive folder ID",
            value: "",
          },
        },
        return: {
          displayName: "Files List",
          type: "list",
          description: "List of files with id, name, mimeType",
        },
      },
      {
        templateId: "moodle.complete_bulk_enrollment_workflow",
        displayName: "Complete Bulk Enrollment Workflow",
        description: "Download Excel from Drive, create students and courses",
        iconCode: "FaUsersCog",
        type: "activity",
        keyword: "Complete Bulk Enrollment Workflow",
        arguments: {
          "Students File ID": {
            type: "string",
            keywordArg: "students_file_id",
            description: "Google Drive file ID for students Excel",
            value: "",
          },
          "Courses File ID": {
            type: "string",
            keywordArg: "courses_file_id",
            description: "Google Drive file ID for courses Excel",
            value: "",
          },
        },
        return: {
          displayName: "Workflow Summary",
          type: "dictionary",
          description: "Complete workflow execution summary",
        },
      },
      // Grading and Reporting
      {
        templateId: "moodle.get_course_grades",
        displayName: "Get Course Grades",
        description: "Get all student grades in a course",
        iconCode: "FaChartBar",
        type: "activity",
        keyword: "Get Course Grades",
        arguments: {
          "Course ID": {
            type: "number",
            keywordArg: "course_id",
            description: "Course ID",
            value: 0,
          },
        },
        return: {
          displayName: "Grades List",
          type: "list",
          description:
            "List of grade dictionaries with student info and grades",
        },
      },
      {
        templateId: "moodle.get_assignment_submissions",
        displayName: "Get Assignment Submissions",
        description: "Get all student submissions for an assignment",
        iconCode: "FaFileAlt",
        type: "activity",
        keyword: "Get Assignment Submissions",
        arguments: {
          "Assignment ID": {
            type: "number",
            keywordArg: "assignment_id",
            description: "Assignment instance ID",
            value: 0,
          },
        },
        return: {
          displayName: "Submissions List",
          type: "list",
          description: "List of submission dictionaries",
        },
      },
      {
        templateId: "moodle.get_assignment_grades",
        displayName: "Get Assignment Grades",
        description: "Get grades for all students in an assignment",
        iconCode: "FaGraduationCap",
        type: "activity",
        keyword: "Get Assignment Grades",
        arguments: {
          "Assignment ID": {
            type: "number",
            keywordArg: "assignment_id",
            description: "Assignment instance ID",
            value: 0,
          },
        },
        return: {
          displayName: "Grades List",
          type: "list",
          description:
            "List of grade dictionaries with username, fullname, grade",
        },
      },
      {
        templateId: "moodle.get_course_enrolled_users",
        displayName: "Get Course Enrolled Users",
        description: "Get all users enrolled in a course",
        iconCode: "FaUsers",
        type: "activity",
        keyword: "Get Course Enrolled Users",
        arguments: {
          "Course ID": {
            type: "number",
            keywordArg: "course_id",
            description: "Course ID",
            value: 0,
          },
        },
        return: {
          displayName: "Users List",
          type: "list",
          description: "List of enrolled user dictionaries",
        },
      },
      {
        templateId: "moodle.export_course_grades_to_excel",
        displayName: "Export Course Grades To Excel",
        description: "Export all course grades to Excel file",
        iconCode: "FaFileExcel",
        type: "activity",
        keyword: "Export Course Grades To Excel",
        arguments: {
          "Course ID": {
            type: "number",
            keywordArg: "course_id",
            description: "Course ID",
            value: 0,
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save Excel file",
            value: "",
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to created Excel file",
        },
      },
      {
        templateId: "moodle.export_course_grades_to_csv",
        displayName: "Export Course Grades To CSV",
        description: "Export all course grades to CSV file",
        iconCode: "FaFileCsv",
        type: "activity",
        keyword: "Export Course Grades To CSV",
        arguments: {
          "Course ID": {
            type: "number",
            keywordArg: "course_id",
            description: "Course ID",
            value: 0,
          },
          "Output Path": {
            type: "string",
            keywordArg: "output_path",
            description: "Path to save CSV file",
            value: "",
          },
        },
        return: {
          displayName: "File Path",
          type: "string",
          description: "Path to created CSV file",
        },
      },
      {
        templateId: "moodle.get_course_activity_completion",
        displayName: "Get Course Activity Completion",
        description: "Get activity completion status for course",
        iconCode: "FaTasks",
        type: "activity",
        keyword: "Get Course Activity Completion",
        arguments: {
          "Course ID": {
            type: "number",
            keywordArg: "course_id",
            description: "Course ID",
            value: 0,
          },
          "User ID": {
            type: "number",
            keywordArg: "userid",
            description: "User ID (optional, leave 0 for all users)",
            value: 0,
          },
        },
        return: {
          displayName: "Completion Data",
          type: "dictionary",
          description: "Activity completion data",
        },
      },
      {
        templateId: "moodle.generate_course_report",
        displayName: "Generate Course Report",
        description:
          "Generate comprehensive course report (grades, enrollments, submissions)",
        iconCode: "FaFileContract",
        type: "activity",
        keyword: "Generate Course Report",
        arguments: {
          "Course ID": {
            type: "number",
            keywordArg: "course_id",
            description: "Course ID",
            value: 0,
          },
          "Output Directory": {
            type: "string",
            keywordArg: "output_dir",
            description: "Directory to save report files",
            value: "./course_reports",
          },
        },
        return: {
          displayName: "Report Files",
          type: "dictionary",
          description: "Dictionary with paths to generated report files",
        },
      },
      {
        templateId: "moodle.get_course_statistics",
        displayName: "Get Course Statistics",
        description: "Get comprehensive statistics about a course",
        iconCode: "FaChartPie",
        type: "activity",
        keyword: "Get Course Statistics",
        arguments: {
          "Course ID": {
            type: "number",
            keywordArg: "course_id",
            description: "Course ID",
            value: 0,
          },
        },
        return: {
          displayName: "Statistics",
          type: "dictionary",
          description:
            "Course statistics including student count, average grade, etc.",
        },
      },
    ],
  },
];
