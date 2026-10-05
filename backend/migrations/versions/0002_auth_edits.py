"""Revocable sessions and summary edits separate from original AI output."""
from alembic import op
import sqlalchemy as sa
revision="0002_auth_edits"
down_revision="0001_meetings_jobs"
branch_labels=None
depends_on=None
def upgrade():
    op.create_table("access_sessions",sa.Column("id",sa.String(64),primary_key=True),sa.Column("username",sa.String(200),nullable=False),sa.Column("role",sa.String(16),nullable=False),sa.Column("expires",sa.Integer(),nullable=False))
    op.create_table("summary_edits",sa.Column("meeting_id",sa.String(160),sa.ForeignKey("meeting_sessions.id",ondelete="CASCADE"),primary_key=True),sa.Column("text",sa.String(50000),nullable=False),sa.Column("updated_at",sa.DateTime(),nullable=False))
def downgrade():
    op.drop_table("summary_edits")
    op.drop_table("access_sessions")
