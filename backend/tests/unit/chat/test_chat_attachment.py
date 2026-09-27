"""Chat attachments stay as files. Their bytes must not become the visible message."""

import io
import unittest
import zipfile

from app.materials.knowledge import parse_bytes
from app.services.chat.session import _compose_turn, prepare_chat_file


def _docx(paragraphs: list[str]) -> bytes:
    # 只放 document.xml。解析器读的就是 w:p / w:t，不需要完整的 Office 包。
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    ).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buf.getvalue()


class ChatAttachmentText(unittest.TestCase):
    def test_docx_is_not_decoded_as_binary(self) -> None:
        data = _docx(["前端开发", "熟悉 React"])
        text = parse_bytes("简历.docx", data)
        self.assertEqual(text, "前端开发\n熟悉 React")
        self.assertNotIn("PK", text)

    def test_prepare_keeps_extracted_text(self) -> None:
        text = prepare_chat_file("简历.docx", _docx(["岗位要求"]))
        self.assertEqual(text, "岗位要求")

    def test_visible_turn_hides_file_body(self) -> None:
        visible, model, files = _compose_turn(
            "帮我看看",
            [{"name": "简历.docx", "size": 12, "text": "熟悉 React"}],
        )
        self.assertEqual(visible, "帮我看看")
        self.assertNotIn("熟悉 React", visible)
        self.assertIn("熟悉 React", model)
        self.assertEqual(files, [{"name": "简历.docx", "size": 12}])

    def test_old_doc_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_bytes("简历.doc", b"\xd0\xcf\x11\xe0")
