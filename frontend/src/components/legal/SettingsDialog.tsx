import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import type { RetrievalSettings } from "@/lib/legal/types";
import { ThemeToggle } from "./Sidebar";

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  settings: RetrievalSettings;
  onSettingsChange: (s: RetrievalSettings) => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export function SettingsDialog({
  open,
  onOpenChange,
  settings,
  onSettingsChange,
  theme,
  onToggleTheme,
}: SettingsDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="font-serif text-[18px]">Cài đặt</DialogTitle>
          <DialogDescription>
            Tùy chỉnh cách hệ thống truy hồi và trình bày căn cứ pháp luật.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5 pt-1">
          <div className="space-y-2">
            <p className="text-[12.5px] font-medium text-foreground">Số lượng nguồn truy hồi</p>
            <div className="flex gap-2">
              {([5, 10, 20] as const).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => onSettingsChange({ ...settings, topK: k })}
                  className={
                    "flex-1 rounded-md border px-3 py-2 text-[13px] font-medium transition-colors " +
                    (settings.topK === k
                      ? "border-[var(--brand)] bg-[color-mix(in_oklab,var(--brand)_18%,transparent)] text-foreground"
                      : "border-border text-muted-foreground hover:bg-accent")
                  }
                >
                  {k} nguồn
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-border pt-4">
            <div>
              <p className="text-[12.5px] font-medium text-foreground">Chế độ nâng cao</p>
              <p className="text-[11.5px] text-muted-foreground">
                Hiển thị điểm dense, sparse, hybrid và rerank
              </p>
            </div>
            <Switch
              checked={settings.advanced}
              onCheckedChange={(v) => onSettingsChange({ ...settings, advanced: v })}
              aria-label="Chế độ nâng cao"
            />
          </div>

          <div className="flex items-center justify-between border-t border-border pt-4">
            <div>
              <p className="text-[12.5px] font-medium text-foreground">Giao diện</p>
              <p className="text-[11.5px] text-muted-foreground">Sáng hoặc tối, được ghi nhớ</p>
            </div>
            <div className="w-40">
              <ThemeToggle theme={theme} onToggle={onToggleTheme} />
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
