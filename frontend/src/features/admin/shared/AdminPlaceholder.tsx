export function AdminPlaceholder({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex h-full flex-col">
      <header className="flex h-12 items-center border-b border-line px-6 text-xs text-ink-2">{title}</header>
      <div className="p-10">
        <AdminPlaceholderNote body={body} />
      </div>
    </div>
  );
}

export function AdminPlaceholderNote({ body }: { body: string }) {
  return <div className="admin-card rounded-xl border border-line bg-card p-4 text-sm leading-6 text-mute">{body}</div>;
}
