"""
tools.py — MCP tool definitions for dev-memory-mcp.

Each tool maps directly onto a developer workflow action.
All tools use Pydantic models for input validation and return
formatted markdown strings for readability in Claude Desktop.

Import and registration pattern:
    from tools import register_tools
    register_tools(mcp)
"""

import json
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

import db
import embeddings

# ── Input models ──────────────────────────────────────────────────────────────

class RememberDecisionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(
        ...,
        description="Project slug (e.g. 'helix', 'free-tier-proxy'). Lowercase, hyphen-separated.",
        min_length=1,
        max_length=80,
    )
    summary: str = Field(
        ...,
        description="One-sentence summary of the decision made (e.g. 'Use SQLite over Postgres for MVP').",
        min_length=1,
        max_length=300,
    )
    reasoning: str = Field(
        ...,
        description="Why this decision was made. The more detail the better — this is what you'll forget.",
        min_length=1,
    )
    alternatives_considered: Optional[list[str]] = Field(
        default_factory=list,
        description="Other options that were considered but not chosen (e.g. ['Postgres', 'MySQL']).",
        max_length=10,
    )


class LogDeadEndInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug.", min_length=1, max_length=80)
    summary: str = Field(
        ...,
        description="One-sentence summary of what was tried (e.g. 'Tried using sqlite-vec with Python 3.11').",
        min_length=1,
        max_length=300,
    )
    what_was_tried: str = Field(
        ...,
        description="Detailed description of the approach attempted.",
        min_length=1,
    )
    why_it_failed: str = Field(
        ...,
        description="Specific reason this approach didn't work.",
        min_length=1,
    )
    conditions_for_retry: Optional[str] = Field(
        default=None,
        description="Under what conditions (if any) this approach might be worth retrying.",
    )


class SaveContextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug.", min_length=1, max_length=80)
    summary: str = Field(
        ...,
        description="Short title for this context snapshot (e.g. 'End of session 2025-03-11').",
        min_length=1,
        max_length=300,
    )
    whats_working: Optional[str] = Field(
        default=None,
        description="What's currently functional or complete.",
    )
    in_progress: Optional[str] = Field(
        default=None,
        description="What's actively being worked on.",
    )
    blocked_on: Optional[str] = Field(
        default=None,
        description="What's blocked and why.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any other freeform notes worth preserving.",
    )


class AddQuestionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug.", min_length=1, max_length=80)
    question: str = Field(
        ...,
        description="A deferred question or open issue to revisit (e.g. 'How should we handle auth token expiry?').",
        min_length=1,
        max_length=500,
    )


class RecallInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(..., description="Project slug to scope the search.", min_length=1, max_length=80)
    query: str = Field(
        ...,
        description="Natural language query (e.g. 'database choice', 'auth approach', 'why did we drop X').",
        min_length=1,
        max_length=300,
    )
    limit: Optional[int] = Field(
        default=8,
        description="Maximum number of results to return.",
        ge=1,
        le=30,
    )


class GetSessionBriefInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    project: str = Field(
        ...,
        description="Project slug to build the session brief for.",
        min_length=1,
        max_length=80,
    )


class ResolveQuestionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    record_id: int = Field(
        ...,
        description="The integer id of the question record to mark as resolved.",
        ge=1,
    )


class ListProjectsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # No fields needed — intentionally empty.


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_record(record: dict) -> str:
    """Render a single record as a markdown block."""
    lines = [
        f"**[{record['type'].upper()} #{record['id']}]** {record['summary']}",
        f"*Project:* `{record['project']}` | *Created:* {record['created_at']}",
    ]

    body = record.get("body")
    if isinstance(body, dict):
        for key, value in body.items():
            if value:
                label = key.replace("_", " ").capitalize()
                if isinstance(value, list):
                    value = ", ".join(value)
                lines.append(f"*{label}:* {value}")

    if record.get("resolved"):
        lines.append("*Status:* ✅ Resolved")

    return "\n".join(lines)


def _fmt_section(title: str, records: list[dict]) -> str:
    """Render a titled section of records, or a 'none found' message."""
    if not records:
        return f"### {title}\n*None found.*"
    body = "\n\n---\n".join(_fmt_record(r) for r in records)
    return f"### {title}\n\n{body}"


# ── Tool registration ─────────────────────────────────────────────────────────

def register_tools(mcp: FastMCP) -> None:
    """Register all dev-memory tools onto the given FastMCP instance."""

    @mcp.tool(
        name="remember_decision",
        annotations={
            "title": "Remember a Decision",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    async def remember_decision(params: RememberDecisionInput) -> str:
        """Record a design or implementation decision with its reasoning.

        Use this whenever a meaningful choice is made during a session —
        architecture, libraries, data models, APIs, etc. The reasoning is
        the most important field; it's what you'll forget first.

        Args:
            params (RememberDecisionInput): Validated input containing:
                - project (str): Project slug
                - summary (str): One-sentence decision summary
                - reasoning (str): Why this decision was made
                - alternatives_considered (list[str]): Options not chosen

        Returns:
            str: Confirmation message with the assigned record id.
        """
        body = {
            "reasoning": params.reasoning,
            "alternatives_considered": params.alternatives_considered or [],
        }
        record_id = db.insert_record(
            project=params.project,
            record_type="decision",
            summary=params.summary,
            body=body,
        )
        embeddings.store_embedding(record_id, embeddings.build_embed_text(params.summary, body))
        return (
            f"✅ Decision recorded (id: **{record_id}**)\n\n"
            f"**{params.summary}**\n"
            f"*Reasoning:* {params.reasoning}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    @mcp.tool(
        name="log_dead_end",
        annotations={
            "title": "Log a Dead End",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    async def log_dead_end(params: LogDeadEndInput) -> str:
        """Record a failed approach so it isn't attempted again.

        This is the most uniquely valuable record type — failed attempts
        rarely make it into documentation, but they're critical context.

        Args:
            params (LogDeadEndInput): Validated input containing:
                - project (str): Project slug
                - summary (str): One-sentence summary of what was tried
                - what_was_tried (str): Detailed description of the approach
                - why_it_failed (str): Specific failure reason
                - conditions_for_retry (Optional[str]): When it might be worth retrying

        Returns:
            str: Confirmation message with the assigned record id.
        """
        body = {
            "what_was_tried": params.what_was_tried,
            "why_it_failed": params.why_it_failed,
            "conditions_for_retry": params.conditions_for_retry,
        }
        record_id = db.insert_record(
            project=params.project,
            record_type="dead_end",
            summary=params.summary,
            body=body,
        )
        embeddings.store_embedding(record_id, embeddings.build_embed_text(params.summary, body))
        return (
            f"🚫 Dead end logged (id: **{record_id}**)\n\n"
            f"**{params.summary}**\n"
            f"*Why it failed:* {params.why_it_failed}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    @mcp.tool(
        name="save_context",
        annotations={
            "title": "Save Session Context",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    async def save_context(params: SaveContextInput) -> str:
        """Snapshot the current state of a project at the end of a session.

        Captures what's working, what's in progress, and what's blocked.
        Retrieve later with get_session_brief to re-orient at the start
        of the next session without rebuilding context from scratch.

        Args:
            params (SaveContextInput): Validated input containing:
                - project (str): Project slug
                - summary (str): Short title for this snapshot
                - whats_working (Optional[str]): Currently functional items
                - in_progress (Optional[str]): Actively worked-on items
                - blocked_on (Optional[str]): What's blocked and why
                - notes (Optional[str]): Freeform notes

        Returns:
            str: Confirmation message with the assigned record id.
        """
        body = {
            "whats_working": params.whats_working,
            "in_progress": params.in_progress,
            "blocked_on": params.blocked_on,
            "notes": params.notes,
        }
        # Strip None values to keep the body clean
        body = {k: v for k, v in body.items() if v is not None}

        record_id = db.insert_record(
            project=params.project,
            record_type="context",
            summary=params.summary,
            body=body,
        )
        embeddings.store_embedding(record_id, embeddings.build_embed_text(params.summary, body))
        return f"💾 Context snapshot saved (id: **{record_id}**)\n\n**{params.summary}**"

    # ──────────────────────────────────────────────────────────────────────────

    @mcp.tool(
        name="add_question",
        annotations={
            "title": "Add an Open Question",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    async def add_question(params: AddQuestionInput) -> str:
        """Record a deferred question or open issue to revisit later.

        Questions that get deferred mid-session typically evaporate.
        This gives them a persistent home. Mark resolved with resolve_question.

        Args:
            params (AddQuestionInput): Validated input containing:
                - project (str): Project slug
                - question (str): The open question or deferred issue

        Returns:
            str: Confirmation message with the assigned record id.
        """
        record_id = db.insert_record(
            project=params.project,
            record_type="question",
            summary=params.question,
        )
        embeddings.store_embedding(record_id, embeddings.build_embed_text(params.question, None))
        return f"❓ Question logged (id: **{record_id}**)\n\n{params.question}"

    # ──────────────────────────────────────────────────────────────────────────

    @mcp.tool(
        name="recall",
        annotations={
            "title": "Recall from Memory",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def recall(params: RecallInput) -> str:
        """Search memory for records relevant to a query.

        Uses semantic vector search (sentence-transformers) to find records
        by meaning, not just keyword match. Works even when exact words differ.

        Use this to answer questions like:
        - 'Have we dealt with auth before?'
        - 'Why did we choose X?'
        - 'What failed when we tried to set up Y?'

        Args:
            params (RecallInput): Validated input containing:
                - project (str): Project slug to scope the search
                - query (str): Natural language search query
                - limit (Optional[int]): Max results (default 8)

        Returns:
            str: Markdown-formatted list of matching records, or a
                 'no results' message.
        """
        results = embeddings.semantic_search(
            project=params.project,
            query=params.query,
            limit=params.limit,
        )

        if not results:
            return f"🔍 No records found in **{params.project}** matching `{params.query}`."

        header = f"🔍 Found **{len(results)}** record(s) in `{params.project}` for `{params.query}`:\n"
        body = "\n\n---\n".join(_fmt_record(r) for r in results)
        return header + "\n" + body

    # ──────────────────────────────────────────────────────────────────────────

    @mcp.tool(
        name="get_session_brief",
        annotations={
            "title": "Get Session Brief",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def get_session_brief(params: GetSessionBriefInput) -> str:
        """Generate a structured brief to orient a new session.

        Call this at the start of a session to instantly restore context:
        recent snapshots, open questions, latest decisions, and known dead ends.
        Eliminates the 'catch me up' problem at the start of every conversation.

        Args:
            params (GetSessionBriefInput): Validated input containing:
                - project (str): Project slug

        Returns:
            str: Markdown-formatted session brief with all key context.
        """
        data = db.get_session_brief_data(params.project)

        if not any(data.values()):
            return f"📋 No records found for project `{params.project}`. Start by saving some context or logging a decision."

        sections = [
            f"# 📋 Session Brief: `{params.project}`\n",
            _fmt_section("💾 Recent Context Snapshots", data["context"]),
            _fmt_section("❓ Open Questions", data["questions"]),
            _fmt_section("✅ Recent Decisions", data["decisions"]),
            _fmt_section("🚫 Known Dead Ends", data["dead_ends"]),
        ]

        return "\n\n".join(sections)

    # ──────────────────────────────────────────────────────────────────────────

    @mcp.tool(
        name="resolve_question",
        annotations={
            "title": "Resolve a Question",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def resolve_question(params: ResolveQuestionInput) -> str:
        """Mark an open question as resolved.

        Use the record id returned when the question was logged,
        or find it via recall or get_session_brief.

        Args:
            params (ResolveQuestionInput): Validated input containing:
                - record_id (int): The question record's integer id

        Returns:
            str: Confirmation, or an error message if the id wasn't found.
        """
        success = db.resolve_question(params.record_id)
        if success:
            return f"✅ Question **#{params.record_id}** marked as resolved."
        return f"⚠️ No unresolved question found with id **#{params.record_id}**. Check the id with recall or get_session_brief."

    # ──────────────────────────────────────────────────────────────────────────

    @mcp.tool(
        name="list_projects",
        annotations={
            "title": "List All Projects",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    async def list_projects(params: ListProjectsInput) -> str:
        """List all projects that have memory records.

        Useful for orienting Claude at the start of a session when
        the project name isn't known.

        Args:
            params (ListProjectsInput): No parameters required.

        Returns:
            str: Bulleted list of project slugs, or a 'no projects' message.
        """
        projects = db.list_projects()
        if not projects:
            return "📂 No projects found. Start by recording a decision or saving context."

        project_list = "\n".join(f"- `{p}`" for p in projects)
        return f"📂 **{len(projects)} project(s) in memory:**\n\n{project_list}"
