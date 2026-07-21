import unittest

from app.services.chunking_service import LegalChunkingService


class LegalChunkingServiceTests(unittest.TestCase):
    def test_preserves_article_and_page_metadata(self) -> None:
        chunks = LegalChunkingService(500, 50).chunk_pages(
            "session",
            "document",
            "luat.txt",
            [
                (1, "CHƯƠNG I\nĐiều 1. Phạm vi\nNội dung trang một."),
                (2, "Nội dung tiếp theo.\nĐiều 2. Đối tượng\nNội dung điều hai."),
            ],
            remove_bare_page_numbers=False,
        )
        self.assertEqual(chunks[0].dieu, "Điều 1")
        self.assertEqual((chunks[0].page_number, chunks[0].page_end_number), (1, 2))
        self.assertEqual(chunks[1].dieu, "Điều 2")
        self.assertEqual(chunks[1].chunk_index, 1)

    def test_splits_long_article_with_overlap(self) -> None:
        text = "Điều 1. Nội dung\n" + " ".join(f"từ{i}" for i in range(300))
        chunks = LegalChunkingService(220, 40).chunk_pages(
            "s", "d", "test.txt", [(1, text)], False
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(item.text) <= 220 for item in chunks))
        self.assertTrue(all(item.content_type == "dieu_partial" for item in chunks))


if __name__ == "__main__":
    unittest.main()
