import markUrl from "../../../../public/generated/sagematch-logo-option-2.png";

/** Shared 知弈 identity mark used by the application shells. */
export function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2" aria-label="知弈">
      <img
        src={markUrl}
        alt=""
        aria-hidden="true"
        className={compact ? "h-7 w-7 rounded-md object-cover" : "h-8 w-8 rounded-md object-cover"}
      />
      {!compact && <span className="text-[15px] font-bold tracking-tight">知弈</span>}
    </div>
  );
}
