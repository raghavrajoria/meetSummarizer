"""Current meeting schema plus persistent jobs; adopt the Milestone 1 table."""
from alembic import op, context
import sqlalchemy as sa
revision = "0001_meetings_jobs"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    existing = False
    if not context.is_offline_mode():
        inspector = sa.inspect(op.get_bind())
        existing = inspector.has_table("meeting_sessions")
        if existing:
            columns = {c["name"] for c in inspector.get_columns("meeting_sessions")}
            if columns != {"id", "title", "group_name", "date", "payload", "media_key"}:
                raise RuntimeError("Existing meeting schema differs; review before migrating")
    if not existing:
        op.create_table("meeting_sessions",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column("title", sa.String(500), nullable=False),
            sa.Column("group_name", sa.String(300), nullable=False),
            sa.Column("date", sa.String(32), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("media_key", sa.String(500), nullable=True))
    op.create_table("jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("meeting_id", sa.String(160), sa.ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("stage", sa.String(40), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("error", sa.String(200), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("worker_token", sa.String(32), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("stage_timings", sa.JSON(), nullable=False),
        sa.CheckConstraint("status IN ('queued','running','done','failed')", name="job_status"),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="job_progress"))
    op.create_index("ix_jobs_status", "jobs", ["status"])

def downgrade():
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_table("jobs")
    # Preserve the original M1 table and recordings, even on a fresh database.
