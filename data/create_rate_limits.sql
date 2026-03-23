-- Run this in the Supabase SQL Editor to create the rate_limits table

create table if not exists rate_limits (
    id bigint primary key generated always as identity,
    ip_or_session text not null,
    timestamp timestamp with time zone default now()
);

-- Index for faster queries
create index if not exists idx_rate_limits_ip on rate_limits(ip_or_session);

-- Enable Row Level Security (RLS) but allow anonymous inserts/selects for the API
alter table rate_limits enable row level security;
create policy "Allow anonymous insert" on rate_limits for insert to anon with check (true);
create policy "Allow anonymous select" on rate_limits for select to anon using (true);
create policy "Allow anonymous delete" on rate_limits for delete to anon using (true);
