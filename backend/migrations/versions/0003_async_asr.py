"""Durable remote ASR checkpoints, including per-participant requests."""
from alembic import op
import sqlalchemy as sa
revision="0003_async_asr"
down_revision="0002_auth_edits"
branch_labels=None
depends_on=None
def upgrade():
    op.add_column("jobs",sa.Column("remote_asr_job_id",sa.String(160),nullable=True))
    op.add_column("jobs",sa.Column("asr_requests",sa.JSON(),nullable=False,server_default="{}"))
def downgrade():
    op.drop_column("jobs","asr_requests")
    op.drop_column("jobs","remote_asr_job_id")
