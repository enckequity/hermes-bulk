"""--over is confined to allowed folders, never the Hermes home; jobs get a default token limit (TeamShift #4654)."""

from __future__ import annotations

from unittest import mock

from bulk_plugin import discovery
from bulk_plugin.commands import BulkCommands
from bulk_plugin.runtime import Runtime

from tests import JobTestCase, StubCtx, open_test_db


class ConfinementTests(JobTestCase):
    def runtime_with(self, home, cwd=None, settings=None, read_block=lambda _path: None) -> BulkCommands:
        runtime = Runtime(
            StubCtx(self.llm, settings),
            data_dir=lambda: self.data_dir,
            open_db=lambda: open_test_db(self.data_dir / "data.db"),
            cwd=lambda: cwd or (self.tmp / "work"),
            sleep=self.sleeps.append,
            hermes_home=lambda: home,
            read_block=read_block,
        )
        self.addCleanup(runtime.close)
        return BulkCommands(runtime)

    def test_path_outside_the_workspace_is_refused(self) -> None:
        outside = self.tmp / "elsewhere"
        outside.mkdir()
        (outside / "a.txt").write_text("secret", encoding="utf-8")
        output = self.commands.handle(f'"x" --over {outside}')
        self.assertIn("--over must be inside", output)
        self.assertIn("must be inside", self.commands.handle('"x" --over ../elsewhere'))

    def test_symbolic_link_out_of_the_workspace_is_refused(self) -> None:
        outside = self.tmp / "elsewhere"
        outside.mkdir()
        (outside / "a.txt").write_text("secret", encoding="utf-8")
        (self.tmp / "work" / "link").symlink_to(outside)
        self.assertIn("must be inside", self.commands.handle('"x" --over link'))

    def test_hermes_home_as_working_directory_is_never_a_root(self) -> None:
        home = self.tmp / "home"
        (home / "workspace" / "docs").mkdir(parents=True)
        (home / "workspace" / "docs" / "a.txt").write_text("row", encoding="utf-8")
        (home / ".env").write_text("KEY=value", encoding="utf-8")
        commands = self.runtime_with(home, cwd=home)
        self.assertIn("must be inside", commands.handle(f'"x" --over {home / ".env"}'))
        self.assertIn("must be inside", commands.handle(f'"x" --over {home}'))
        # A relative path resolves inside <home>/workspace, which stays allowed.
        self.assertIn("with 1 item", commands.handle('"x" --over docs'))

    def test_a_folder_containing_the_home_is_never_a_root(self) -> None:
        home = self.tmp / "work" / "data"
        home.mkdir()
        commands = self.runtime_with(home, settings={"allowed_roots": [str(self.tmp / "work")]})
        self.assertIn("no folder is open", commands.handle('"x" --over reports'))

    def test_configured_roots_replace_the_default(self) -> None:
        shared = self.tmp / "shared"
        shared.mkdir()
        (shared / "a.txt").write_text("row", encoding="utf-8")
        commands = self.runtime_with(self.tmp / "home", settings={"allowed_roots": [str(shared), "relative/ignored"]})
        self.assertIn("with 1 item", commands.handle('"x" --over a.txt'))
        self.assertIn("must be inside", commands.handle(f'"x" --over {self.reports}'))

    def test_protected_source_is_refused(self) -> None:
        commands = self.runtime_with(self.tmp / "home", read_block=lambda path: "blocked" if path.endswith("q1.txt") else None)
        output = commands.handle(f'"x" --over {self.reports / "q1.txt"}')
        self.assertIn("protected file", output)
        self.assertNotIn("blocked", output)

    def test_protected_files_in_a_folder_are_skipped(self) -> None:
        with mock.patch.object(discovery, "_read_block", lambda path: "blocked" if path.endswith("q1.txt") else None):
            output = self.commands.handle('"x" --over reports')
        self.assertIn("with 1 item", output)
        self.assertIn("q1.txt: q1.txt is a protected file", output)

    def test_default_token_limit_and_concurrency(self) -> None:
        job_id = self.create_job()
        self.assertEqual(self.job(job_id)["max_tokens"], 200_000)
        self.assertEqual(self.runtime.max_concurrency(), 4)
        runtime = self.make_runtime(self.llm, settings={"default_max_tokens": 5000})
        self.addCleanup(runtime.close)
        output = BulkCommands(runtime).handle('"x" --over reports')
        self.assertEqual(runtime.manager().store.get_job(output.split()[2])["max_tokens"], 5000)
        explicit = self.create_job("--max-tokens 999")
        self.assertEqual(self.job(explicit)["max_tokens"], 999)

    def test_status_text_names_no_host(self) -> None:
        job_id = self.create_job()
        text = self.commands.handle(f"status {job_id}") + self.commands.handle("help")
        self.assertNotIn("Hermes", text)
