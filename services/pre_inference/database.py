import os
import json
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

# Initialize database URL from environment variable, replacing postgresql:// with postgresql+asyncpg:// if needed
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# IMPORTANT: When using Supabase IPv4 Pooler (PgBouncer in transaction mode),
# asyncpg will crash with DuplicatePreparedStatementError unless we disable the statement cache!
engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    connect_args={
        "prepared_statement_cache_size": 0,
        "statement_cache_size": 0
    }
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

class Tier1Prediction(Base):
    __tablename__ = "tier1_predictions"

    # We use image_id as a primary key, or an auto-incrementing ID. 
    # The requirement says "image_id: str = Field(..., description='Stable UUID assigned at ingestion')"
    # We can use a surrogate primary key or image_id. Let's use an auto-incrementing id and index image_id.
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(index=True)
    image_url: Mapped[str] = mapped_column()
    text_prompt: Mapped[str] = mapped_column()
    
    bounding_box: Mapped[dict] = mapped_column(JSONB)
    mcd_samples: Mapped[list] = mapped_column(JSONB)
    
    geometric_variance: Mapped[float] = mapped_column()
    class_logit_entropy: Mapped[float] = mapped_column()
    consensus_mask: Mapped[dict] = mapped_column(JSONB)
    
    model_version: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
